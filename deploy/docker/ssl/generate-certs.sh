#!/bin/bash

# Create SSL certificates directory
mkdir -p /etc/ssl/certs /etc/ssl/private

# Generate private key
openssl genrsa -out /etc/ssl/private/key.pem 2048

# Generate self-signed certificate
openssl req -new -x509 -key /etc/ssl/private/key.pem -out /etc/ssl/certs/cert.pem -days 365 -subj "/C=US/ST=CA/L=SF/O=RealTimeTranslator/CN=localhost"

# Set proper permissions
chmod 600 /etc/ssl/private/key.pem
chmod 644 /etc/ssl/certs/cert.pem

echo "SSL certificates generated successfully!"
echo "Certificate: /etc/ssl/certs/cert.pem"
echo "Private key: /etc/ssl/private/key.pem"
