import json
import yaml
import csv
import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# PDF and Excel parsing libraries
try:
    import PyPDF2
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


class SchemaParser:
    """
    Enhanced schema parser that normalizes class/topic schema files into
    standardized expected_topics.json format.
    """

    def __init__(self, output_dir: str = "schemas/normalized"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_schema(self, file_path: str) -> Dict[str, Any]:
        """
        Main entry point for parsing schema files.
        Supports JSON, YAML, CSV, PDF, and Excel formats.
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Schema file not found: {file_path}")

        suffix = file_path.suffix.lower()

        if suffix == '.json':
            return self._parse_json_schema(file_path)
        elif suffix in ['.yaml', '.yml']:
            return self._parse_yaml_schema(file_path)
        elif suffix == '.csv':
            return self._parse_csv_schema(file_path)
        elif suffix == '.pdf':
            return self._parse_pdf_schema(file_path)
        elif suffix in ['.xlsx', '.xls']:
            return self._parse_excel_schema(file_path)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. Supported formats: JSON, YAML, CSV, PDF, Excel")

    def _parse_json_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse JSON schema file with validation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema = json.load(f)
            return self._normalize_schema(schema, file_path.stem)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {file_path}: {e}")

    def _parse_yaml_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse YAML schema file with validation."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                schema = yaml.safe_load(f)
            return self._normalize_schema(schema, file_path.stem)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format in {file_path}: {e}")

    def _parse_csv_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse CSV schema file with flexible column handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                # Try to detect delimiter
                sample = f.read(1024)
                f.seek(0)
                sniffer = csv.Sniffer()
                delimiter = sniffer.sniff(sample).delimiter

                reader = csv.DictReader(f, delimiter=delimiter)
                rows = list(reader)

                if not rows:
                    raise ValueError("CSV file is empty")

                # Handle multiple rows (classes) or single row
                if len(rows) == 1:
                    return self._normalize_csv_row(rows[0], file_path.stem)
                else:
                    # For multiple classes, return the first one or merge logic
                    return self._normalize_csv_row(rows[0], file_path.stem)

        except Exception as e:
            raise ValueError(f"Error parsing CSV file {file_path}: {e}")

    def _normalize_csv_row(self, row: Dict[str, str], filename: str) -> Dict[str, Any]:
        """Normalize a single CSV row into standard format."""
        # Flexible field mapping
        field_mappings = {
            'class_id': ['class_id', 'classid', 'class', 'id'],
            'date': ['date', 'class_date', 'session_date'],
            'expected_topics': ['expected_topics', 'topics', 'curriculum', 'syllabus']
        }

        normalized = {}

        # Extract class_id
        class_id = self._find_field_value(row, field_mappings['class_id'])
        normalized['class_id'] = class_id or filename

        # Extract date
        date_str = self._find_field_value(row, field_mappings['date'])
        normalized['date'] = self._parse_date(date_str)

        # Extract topics
        topics_str = self._find_field_value(row, field_mappings['expected_topics'])
        normalized['expected_topics'] = self._parse_topics_string(topics_str)

        return normalized

    def _parse_pdf_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse PDF schema files using PyPDF2."""
        if not PDF_AVAILABLE:
            raise ImportError("PyPDF2 is required for PDF parsing but not installed.")

        try:
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""

            # Find the fields using simple regex
            class_id = self._find_field_from_text(
                text, ["class_id", "classid", "Course Code"])
            date_str = self._find_field_from_text(
                text, ["date", "class_date", "session_date"])
            topics_str = self._find_field_from_text(
                text, ["topics", "expected_topics", "syllabus", "curriculum"])

            normalized = {
                'class_id': class_id or file_path.stem,
                'date': self._parse_date(date_str),
                'expected_topics': self._parse_topics_string(topics_str),
                'metadata': {'original_format': 'pdf', 'source_file': file_path.name}
            }

            return normalized
        except Exception as e:
            raise ValueError(f"Error parsing PDF file {file_path}: {e}")

    def _parse_excel_schema(self, file_path: Path) -> Dict[str, Any]:
        """Parse Excel schema files using pandas."""
        if not PANDAS_AVAILABLE:
            raise ImportError(
                "pandas is required for Excel parsing but not installed.")

        try:
            df = pd.read_excel(file_path)

            if df.empty:
                raise ValueError("Excel file is empty")

            # Assuming similar structure as CSV handling
            row = df.iloc[0].to_dict()
            return self._normalize_csv_row(row, file_path.stem)
        except Exception as e:
            raise ValueError(f"Error parsing Excel file {file_path}: {e}")

    def _find_field_from_text(self, text: str, possible_keys: List[str]) -> Optional[str]:
        """Search for given field names in text and extract value."""
        # Simple method using regex search for keys
        for key in possible_keys:
            pattern = rf"\b{key}:?\s*(.+)\b"
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        return None

    def _find_field_value(self, row: Dict[str, str], possible_keys: List[str]) -> Optional[str]:
        """Find value from row using possible key variations."""
        for key in possible_keys:
            # Try exact match first
            if key in row and row[key].strip():
                return row[key].strip()

        # Try case-insensitive match
        for row_key in row.keys():
            if row_key.lower() == key.lower() and row[row_key].strip():
                return row[row_key].strip()
        return None

    def _normalize_schema(self, schema: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Normalize schema data into standard format."""
        normalized = {
            'class_id': schema.get('class_id') or schema.get('classId') or filename,
            'date': self._parse_date(schema.get('date')),
            'expected_topics': self._normalize_topics(schema.get('expected_topics', [])),
            'metadata': {
                'original_format': 'json' if isinstance(schema, dict) else 'yaml',
                'parsed_at': datetime.now().isoformat(),
                'source_file': filename
            }
        }

        # Add optional fields if present
        optional_fields = ['course_name', 'instructor', 'duration', 'level']
        for field in optional_fields:
            if field in schema:
                normalized['metadata'][field] = schema[field]

        return normalized

    def _parse_date(self, date_input: Any) -> str:
        """Parse and standardize date format."""
        if not date_input:
            return datetime.now().strftime('%Y-%m-%d')

        if isinstance(date_input, str):
            # Try common date formats
            date_formats = ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y-%m-%d %H:%M:%S']
            for fmt in date_formats:
                try:
                    parsed_date = datetime.strptime(date_input, fmt)
                    return parsed_date.strftime('%Y-%m-%d')
                except ValueError:
                    continue

            # If no format matches, use current date
            return datetime.now().strftime('%Y-%m-%d')

        return str(date_input)

    def _normalize_topics(self, topics: Any) -> List[str]:
        """Normalize topics into a clean list of strings."""
        if isinstance(topics, str):
            return self._parse_topics_string(topics)
        elif isinstance(topics, list):
            return [str(topic).strip() for topic in topics if str(topic).strip()]
        else:
            return []

    def _parse_topics_string(self, topics_str: str) -> List[str]:
        """Parse topics from a string with various delimiters."""
        if not topics_str:
            return []

        # Try different delimiters
        delimiters = [',', ';', '|', '\n']
        for delimiter in delimiters:
            if delimiter in topics_str:
                topics = [topic.strip() for topic in topics_str.split(delimiter)]
                return [topic for topic in topics if topic]

        # If no delimiter found, treat as single topic
        return [topics_str.strip()] if topics_str.strip() else []

    def save_normalized_schema(self, normalized_data: Dict[str, Any]) -> str:
        """Save normalized schema to expected_topics.json format."""
        class_id = normalized_data['class_id']
        date = normalized_data['date']

        filename = f"{class_id}_{date}_expected_topics.json"
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(normalized_data, f, indent=2, ensure_ascii=False)

        return str(output_path)

    def process_schema_file(self, input_file: str) -> str:
        """Complete pipeline: parse and save normalized schema."""
        normalized_data = self.parse_schema(input_file)
        output_path = self.save_normalized_schema(normalized_data)
        return output_path


# Convenience functions for backward compatibility

def parse_schema(file_path: str) -> Dict[str, Any]:
    """Parse schema file and return normalized data."""
    parser = SchemaParser()
    return parser.parse_schema(file_path)


def create_expected_topics_json(
        input_file: str, output_dir: str = "schemas/normalized") -> str:
    """Create normalized expected_topics.json from input schema file."""
    parser = SchemaParser(output_dir)
    return parser.process_schema_file(input_file)


def main():
    """
    Command-line interface for schema parsing and normalization.
    """
    parser = argparse.ArgumentParser(
        description='Parse class/topic schema files and convert to normalized expected_topics.json format.'
    )
    parser.add_argument(
        'input_file',
        help='Path to input schema file (JSON, YAML, or CSV)')
    parser.add_argument('--output-dir', '-o', default='schemas/normalized',
                        help='Output directory for normalized files (default: schemas/normalized)')
    parser.add_argument('--validate', '-v', action='store_true',
                        help='Validate the parsed schema')

    args = parser.parse_args()

    try:
        schema_parser = SchemaParser(args.output_dir)
        output_path = schema_parser.process_schema_file(args.input_file)

        print(f"[PASS] Successfully parsed and normalized schema")
        print(f" Input file: {args.input_file}")
        print(f" Output file: {output_path}")

        if args.validate:
            # Load and display the normalized data
            with open(output_path, 'r') as f:
                data = json.load(f)
            print(f"\nNormalized Schema:")
            print(json.dumps(data, indent=2))

    except Exception as e:
        print(f"[FAIL] Error processing schema file: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
