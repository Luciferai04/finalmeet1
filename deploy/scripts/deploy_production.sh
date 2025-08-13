#!/bin/bash
# Production deployment script for Real-Time Translator

set -e

echo "Starting Production Deployment"
echo "=================================="

# Configuration
COMPOSE_FILE="docker-compose.prod.yml"
BACKUP_DIR="backups"
DEPLOY_LOG="logs/deployment.log"

# Create logs directory
mkdir -p logs

# Function to log messages
log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a "$DEPLOY_LOG"
}

# Function to check if Docker is running
check_docker() {
    log "Checking Docker status..."
    if ! docker info >/dev/null 2>&1; then
        log "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log "Docker is running"
}

# Function to check required files
check_files() {
    log "Checking required files..."
    
    required_files=("$COMPOSE_FILE" "Dockerfile.prod" "flask_api_fixed.py" ".env")
    
    for file in "${required_files[@]}"; do
        if [[ ! -f "$file" ]]; then
            log "Required file not found: $file"
            exit 1
        fi
    done
    
    log "All required files found"
}

# Function to create backup
create_backup() {
    log "Creating backup before deployment..."
    
    if [[ -x "./backup.sh" ]]; then
        ./backup.sh
        log "Backup created successfully"
    else
        log "Backup script not found or not executable"
    fi
}

# Function to generate SSL certificates if needed
setup_ssl() {
    log "Setting up SSL certificates..."
    
    if [[ ! -f "ssl/private.key" ]] || [[ ! -f "ssl/certificate.crt" ]]; then
        if [[ -x "./generate_ssl_certs.sh" ]]; then
            ./generate_ssl_certs.sh
            log "SSL certificates generated"
        else
            log "SSL certificate script not found"
        fi
    else
        log "SSL certificates already exist"
    fi
}

# Function to pull latest images
pull_images() {
    log "Pulling latest Docker images..."
    docker-compose -f "$COMPOSE_FILE" pull
    log "Images pulled successfully"
}

# Function to build and deploy
deploy() {
    log "Building and deploying services..."
    
    # Stop existing services
    log "Stopping existing services..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    
    # Build and start services
    log "Starting new services..."
    docker-compose -f "$COMPOSE_FILE" up -d --build
    
    log "Services deployed successfully"
}

# Function to wait for services to be healthy
wait_for_services() {
    log "Waiting for services to be healthy..."
    
    max_attempts=30
    attempt=0
    
    while [[ $attempt -lt $max_attempts ]]; do
        if curl -s http://localhost:8000/api/health >/dev/null 2>&1; then
            log "Application is healthy"
            break
        fi
        
        attempt=$((attempt + 1))
        log "Attempt $attempt/$max_attempts - waiting for application..."
        sleep 10
    done
    
    if [[ $attempt -eq $max_attempts ]]; then
        log "Application failed to become healthy"
        log "Checking service status..."
        docker-compose -f "$COMPOSE_FILE" ps
        exit 1
    fi
}

# Function to run health checks
health_check() {
    log "Running health checks..."
    
    # Check main application
    if curl -s http://localhost:8000/api/health >/dev/null; then
        log "Main application: healthy"
    else
        log "Main application: unhealthy"
        return 1
    fi
    
    # Check Prometheus
    if curl -s http://localhost:9090/-/healthy >/dev/null; then
        log "Prometheus: healthy"
    else
        log "Prometheus: not accessible"
    fi
    
    # Check Grafana
    if curl -s http://localhost:3000/api/health >/dev/null; then
        log "Grafana: healthy"
    else
        log "Grafana: not accessible"
    fi
    
    log "Health checks completed"
}

# Function to display deployment info
show_info() {
    log "Deployment Information"
    log "=========================="
    log "Application: http://localhost:8000"
    log "Prometheus: http://localhost:9090"
    log "Grafana: http://localhost:3000 (admin/admin123)"
    log "Compose file: $COMPOSE_FILE"
    log "Logs: $DEPLOY_LOG"
    log ""
    log "Running containers:"
    docker-compose -f "$COMPOSE_FILE" ps
}

# Function to cleanup on failure
cleanup_on_failure() {
    log "Deployment failed. Cleaning up..."
    docker-compose -f "$COMPOSE_FILE" down --remove-orphans
    log "Cleanup completed"
}

# Main execution
main() {
    log "Starting production deployment process..."
    
    # Set trap for cleanup on failure
    trap cleanup_on_failure ERR
    
    # Run deployment steps
    check_docker
    check_files
    create_backup
    setup_ssl
    pull_images
    deploy
    wait_for_services
    health_check
    show_info
    
    log "Production deployment completed successfully!"
    log "Your Real-Time Translator is now running in production mode"
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    stop)
        log "Stopping production services..."
        docker-compose -f "$COMPOSE_FILE" down
        log "Services stopped"
        ;;
    restart)
        log "Restarting production services..."
        docker-compose -f "$COMPOSE_FILE" restart
        log "Services restarted"
        ;;
    status)
        log "Service status:"
        docker-compose -f "$COMPOSE_FILE" ps
        ;;
    logs)
        docker-compose -f "$COMPOSE_FILE" logs -f
        ;;
    *)
        echo "Usage: $0 {deploy|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  deploy  - Deploy the application (default)"
        echo "  stop    - Stop all services"
        echo "  restart - Restart all services"
        echo "  status  - Show service status"
        echo "  logs    - Show service logs"
        exit 1
        ;;
esac
