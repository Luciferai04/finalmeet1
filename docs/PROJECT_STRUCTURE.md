# Real-Time Translator - Project Structure

This document describes the production-ready project structure for the Real-Time Translator application.

## Directory Structure

```
real-time-translator/
├── README.md                          # Main project documentation
├── requirements.txt                   # Python dependencies
├── main.py                           # Main application entry point
├── wsgi.py                           # WSGI entry point for production
├── run_ui.py                         # Gradio UI launcher
├── gunicorn.conf.py                  # Gunicorn configuration
│
├── src/                              # Source code
│   ├── __init__.py
│   ├── api/                          # API endpoints and routes
│   │   ├── __init__.py
│   │   ├── flask_api_fixed.py        # Main Flask API
│   │   └── routes.py                 # API route definitions
│   ├── core/                         # Core application logic
│   │   ├── __init__.py
│   │   ├── app_factory.py            # Application factory
│   │   ├── config.py                 # Configuration management
│   │   ├── logging_config.py         # Logging configuration
│   │   └── production_config.py      # Production-specific config
│   ├── services/                     # Business logic services
│   │   ├── __init__.py
│   │   ├── rl_coordinator.py         # Reinforcement learning coordinator
│   │   ├── translation_service.py    # Translation services
│   │   ├── whisper_live/             # WhisperLive integration
│   │   ├── schema_checker/           # Schema checking services
│   │   └── ... (other services)
│   ├── ui/                          # User interface
│   │   ├── __init__.py
│   │   └── live_camera_enhanced_ui.py # Gradio UI implementation
│   └── utils/                       # Utility functions
│       ├── __init__.py
│       └── validators.py            # Input validation utilities
│
├── config/                          # Configuration files
│   ├── config.json                  # Base configuration
│   ├── gunicorn.conf.py            # Gunicorn settings
│   ├── .env                        # Environment variables (development)
│   ├── .env.ssl                    # SSL configuration
│   ├── environments/               # Environment-specific configs
│   │   ├── development.env
│   │   └── production.env
│   ├── nginx/                      # Nginx configuration
│   └── ssl/                        # SSL certificates
│
├── deploy/                         # Deployment configurations
│   ├── docker/                     # Docker configurations
│   │   ├── docker-compose.yml
│   │   ├── docker-compose.prod.yml
│   │   ├── Dockerfile
│   │   └── Dockerfile.prod
│   ├── kubernetes/                 # Kubernetes manifests
│   ├── cloud/                      # Cloud deployment configs
│   │   ├── aws/
│   │   ├── gcp/
│   │   └── azure/
│   └── scripts/                    # Deployment scripts
│       └── deploy_production.sh
│
├── data/                           # Data and runtime files
│   ├── uploads/                    # File uploads
│   ├── reports/                    # Generated reports
│   ├── transcripts/                # Session transcripts
│   ├── schemas/                    # Schema files
│   ├── course_materials/           # Course materials
│   ├── logs/                       # Application logs
│   └── monitoring/                 # Monitoring configurations
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── unit/                       # Unit tests
│   ├── integration/                # Integration tests
│   └── e2e/                        # End-to-end tests
│
├── docs/                           # Documentation
│   ├── PROJECT_STRUCTURE.md        # This file
│   ├── API.md                      # API documentation
│   ├── DEPLOYMENT.md               # Deployment guide
│   └── DEVELOPMENT.md              # Development guide
│
└── WhisperLive/                    # WhisperLive submodule/dependency
```

## Key Components

### 1. Source Code (`src/`)
- **api/**: REST API endpoints and Flask application
- **core/**: Core application logic, configuration, and factories
- **services/**: Business logic services (translation, AI, etc.)
- **ui/**: User interface components (Gradio-based)
- **utils/**: Shared utility functions

### 2. Configuration (`config/`)
- Environment-specific configuration files
- SSL certificates and security settings
- Server configurations (Nginx, Gunicorn)

### 3. Deployment (`deploy/`)
- Docker containers and compose files
- Kubernetes manifests
- Cloud deployment configurations
- Deployment automation scripts

### 4. Data (`data/`)
- Runtime data storage
- File uploads and processing
- Generated reports and transcripts
- Application logs

## Entry Points

### Development
```bash
# Flask development server
python main.py

# Gradio UI
python run_ui.py
```

### Production
```bash
# Using Gunicorn
gunicorn --config gunicorn.conf.py wsgi:app

# Using Docker
docker-compose -f deploy/docker/docker-compose.prod.yml up
```

## Environment Configuration

### Development
- Load from `config/environments/development.env`
- Debug mode enabled
- Detailed logging
- Auto-reload on code changes

### Production
- Load from `config/environments/production.env`
- Optimized for performance
- Security hardened
- Monitoring enabled

## Security Considerations

1. **Environment Variables**: Sensitive data stored in environment variables
2. **SSL/TLS**: HTTPS enabled in production
3. **Input Validation**: All inputs validated and sanitized
4. **Rate Limiting**: API rate limiting implemented
5. **Authentication**: Secure authentication mechanisms

## Monitoring and Logging

- **Application Logs**: Structured logging in `data/logs/`
- **Performance Monitoring**: Prometheus metrics
- **Health Checks**: Built-in health check endpoints
- **Error Tracking**: Comprehensive error handling and reporting

## Data Flow

1. **User Input** → UI/API endpoints
2. **Processing** → Services (translation, transcription)
3. **Storage** → Data directory (transcripts, reports)
4. **Response** → UI/API response

This structure ensures:
-  Clear separation of concerns
-  Scalable architecture
-  Production readiness
-  Easy maintenance and deployment
-  Comprehensive testing support
