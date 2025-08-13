"""
Schema Service for handling schema-related operations
"""

import os
import json
from typing import Dict, List, Any, Optional
from datetime import datetime


class SchemaService:
    """Service for managing schema operations"""
    
    def __init__(self, schema_dir: str = "schemas"):
        self.schema_dir = schema_dir
        os.makedirs(schema_dir, exist_ok=True)
        os.makedirs(os.path.join(schema_dir, "normalized"), exist_ok=True)
    
    def create_schema(self, class_id: str, topics: List[str], metadata: Dict[str, Any] = None) -> str:
        """Create a new schema file"""
        schema_data = {
            "class_id": class_id,
            "date": datetime.now().strftime("%Y-%m-%d"),
            "expected_topics": topics,
            "metadata": metadata or {}
        }
        
        schema_path = os.path.join(self.schema_dir, "normalized", f"{class_id}_schema.json")
        with open(schema_path, 'w', encoding='utf-8') as f:
            json.dump(schema_data, f, indent=2, ensure_ascii=False)
        
        return schema_path
    
    def load_schema(self, schema_path: str) -> Optional[Dict[str, Any]]:
        """Load schema from file"""
        try:
            with open(schema_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading schema: {e}")
            return None
    
    def list_schemas(self) -> List[str]:
        """List all available schema files"""
        schema_files = []
        normalized_dir = os.path.join(self.schema_dir, "normalized")
        
        if os.path.exists(normalized_dir):
            for file in os.listdir(normalized_dir):
                if file.endswith('.json'):
                    schema_files.append(os.path.join(normalized_dir, file))
        
        return schema_files
    
    def validate_schema(self, schema_data: Dict[str, Any]) -> bool:
        """Validate schema structure"""
        required_fields = ['class_id', 'expected_topics']
        return all(field in schema_data for field in required_fields)
    
    def compare_schemas(self, schema1_path: str, schema2_path: str) -> Dict[str, Any]:
        """Compare two schemas"""
        schema1 = self.load_schema(schema1_path)
        schema2 = self.load_schema(schema2_path)
        
        if not schema1 or not schema2:
            return {"error": "Could not load one or both schemas"}
        
        topics1 = set(schema1.get('expected_topics', []))
        topics2 = set(schema2.get('expected_topics', []))
        
        return {
            "common_topics": list(topics1.intersection(topics2)),
            "unique_to_first": list(topics1 - topics2),
            "unique_to_second": list(topics2 - topics1),
            "similarity_score": len(topics1.intersection(topics2)) / len(topics1.union(topics2)) if topics1.union(topics2) else 0
        }
