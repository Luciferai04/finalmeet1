# Real-Time Translator with Advanced Translation Engine

## Overview

The Real-Time Translator is a state-of-the-art application for real-time speech translation with advanced AI capabilities. It features a next-generation **Advanced Translation Engine** with context awareness, quality assessment, and cultural adaptation, powered by Google Gemini AI and WhisperLive for seamless real-time translation experiences.

## 🚀 Key Features

### **Advanced Translation Engine**
- **Context-Aware Translation**: Maintains conversation history and adapts to context
- **Domain Detection**: Automatically detects and adapts to business, technical, medical, legal, and educational content
- **Cultural Sensitivity**: Bengali and Hindi cultural context with proper honorifics and regional variations
- **Quality Assessment**: Real-time translation quality scoring and iterative improvements
- **Multi-Stage Pipeline**: Pre-processing, translation, post-processing, and quality validation
- **Terminology Consistency**: Maintains consistent translation of technical terms across sessions

### **Real-Time Processing**
- **WhisperLive Integration**: Live speech-to-text transcription with voice activity detection
- **Live Camera Feed**: Real-time webcam integration for immersive translation experience
- **Streaming Translation**: Instant translation display as you speak
- **Session Management**: Complete session recording with transcript generation

### **Intelligent Features**
- **Metacognitive Controller**: Adaptive strategy selection based on content type
- **EgoSchema Integration**: Advanced video understanding and evaluation
- **Performance Monitoring**: Real-time system performance tracking and optimization
- **Topic Analysis**: Course material comparison against live transcripts

### **Production-Ready Architecture**
- **Robust Error Handling**: Automatic reconnection and graceful degradation
- **Scalable Design**: Modular architecture with microservices support
- **Monitoring & Logging**: Comprehensive system monitoring with Prometheus/Grafana
- **Deployment Options**: Docker, Kubernetes, and cloud platform support

## Directory Structure

For detailed information, refer to the [Project Structure Documentation](docs/PROJECT_STRUCTURE.md).

## Getting Started

### Prerequisites

- Python 3.8+
- Docker & Docker Compose
- Google Gemini API key

### Development Setup

1. **Clone the Repository**:
    ```bash
git clone https://github.com/Luciferai04/finalmeet1.git
cd finalmeet1
    ```

2. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Install WhisperLive**:
    ```bash
    # Install WhisperLive from GitHub
    pip install git+https://github.com/collabora/WhisperLive.git
    
    # Or clone and install locally
    git clone https://github.com/collabora/WhisperLive.git
    cd WhisperLive
    pip install -e .
    cd ..
    ```

4. **Configure Environment**:
    ```bash
    # Copy example environment file
    cp config/environments/development.env.example config/environments/development.env
    
    # Edit the file and add your Google API key
    export GOOGLE_API_KEY="your-google-gemini-api-key-here"
    ```

5. **Run the Application**:
    ```bash
    python main.py
    ```

### Running the Gradio UI

1. **Start WhisperLive Server**:
    ```bash
    # In a separate terminal, start the WhisperLive server
    python -m whisper_live.server --port 9090
    
    # Or if you cloned WhisperLive locally:
    cd WhisperLive
    python run_server.py --port 9090
    cd ..
    ```

2. **Optional: Start Redis** (for advanced features):
    ```bash
    redis-server
    ```

3. **Launch the UI**:
    ```bash
    python run_ui.py
    ```
    
    The interface will be available at: `http://localhost:7860`

### Configuration

- Configuration is managed via environment variables and dotenv files located in `config/environments/`.
- Copy the example files and customize them:
  ```bash
  # Development configuration
  cp config/environments/development.env.example config/environments/development.env
  
  # Production configuration  
  cp config/environments/production.env.example config/environments/production.env
  ```
- **Required**: Set your Google Gemini API key in the environment file or as an environment variable.
- For SSL/HTTPS setup, see `config/ssl/README.md` for certificate placement instructions.

### Deployment

We provide several methods for deployment, including Docker and direct server deployment with Gunicorn.

- **Docker**:
    ```bash
    docker-compose -f deploy/docker/docker-compose.prod.yml up --build
    ```

- **Gunicorn**:
    ```bash
    gunicorn --config config/gunicorn.conf.py wsgi:app
    ```

For more information, please refer to the [Deployment Guide](docs/DEPLOYMENT.md).

## Security Practices

- All sensitive data is stored securely using environment variables.
- SSL/TLS is enabled for secure communication.
- Input validation and sanitization are enforced on all endpoints.
- Rate limiting is applied to API requests to protect against abuse.

## Monitoring and Maintenance

- **Logging**: Logs are stored in the `data/logs/` directory.
- **Monitoring**: Prometheus and Grafana can be configured for real-time monitoring.
- **Error Tracking**: Detailed error handling and reporting are in place to track and resolve issues.

## Contribution

We welcome contributions! Before submitting a pull request, ensure that your changes are covered by tests where applicable, and that they follow the project's coding conventions.

## License

This project is licensed under the MIT License.
