"""
Gunicorn Configuration for Real-Time Translator
===============================================

Production-ready configuration for running the application with Gunicorn.
"""

import os
import multiprocessing

# Server socket
bind = f"0.0.0.0:{os.environ.get('PORT', 8000)}"
backlog = 2048

# Worker processes
workers = int(os.environ.get('GUNICORN_WORKERS', multiprocessing.cpu_count() * 2 + 1))
worker_class = "sync"
worker_connections = 1000
timeout = int(os.environ.get('GUNICORN_TIMEOUT', 300))
keepalive = 2

# Restart workers after this many requests, to help prevent memory leaks
max_requests = 1000
max_requests_jitter = 100

# Logging
accesslog = os.environ.get('GUNICORN_ACCESS_LOG', 'data/logs/gunicorn-access.log')
errorlog = os.environ.get('GUNICORN_ERROR_LOG', 'data/logs/gunicorn-error.log')
loglevel = os.environ.get('GUNICORN_LOG_LEVEL', 'info')
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Process naming
proc_name = 'real-time-translator'

# Server mechanics
daemon = False
pidfile = '/tmp/gunicorn.pid'
user = None
group = None
tmp_upload_dir = None

# SSL (if using HTTPS)
keyfile = os.environ.get('SSL_KEYFILE')
certfile = os.environ.get('SSL_CERTFILE')

# Preload application for better performance
preload_app = True

# Enable automatic restarts when code changes (development only)
if os.environ.get('FLASK_ENV') == 'development':
    reload = True
    reload_extra_files = []

def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Real-Time Translator server...")

def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading Real-Time Translator server...")

def worker_int(worker):
    """Called just after a worker has been interrupted."""
    worker.log.info("Worker %s interrupted", worker.pid)

def on_exit(server):
    """Called just before the master process exits."""
    server.log.info("Shutting down Real-Time Translator server...")

# Environment-specific configurations
if os.environ.get('FLASK_ENV') == 'production':
    # Production settings
    workers = int(os.environ.get('GUNICORN_WORKERS', 4))
    timeout = 300
    loglevel = 'warning'
elif os.environ.get('FLASK_ENV') == 'development':
    # Development settings
    workers = 1
    timeout = 120
    loglevel = 'debug'
    reload = True
