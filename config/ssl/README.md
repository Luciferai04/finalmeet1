# SSL Certificate Configuration

This directory is for SSL/TLS certificates used in production deployments.

## Required Files for SSL/HTTPS

Place the following certificate files in this directory for production use:

### Certificate Files
- `fullchain.pem` - Full certificate chain (certificate + intermediate certificates)
- `cert.pem` - Server certificate only
- `private.key` - Private key file
- `key.pem` - Alternative private key file name

### Optional Files
- `certificate.crt` - Certificate in CRT format
- `certificate.csr` - Certificate signing request (for renewals)

## File Permissions

Ensure proper file permissions for security:

```bash
# Certificate files (readable by all)
chmod 644 *.pem *.crt *.csr

# Private key files (readable only by owner)
chmod 600 *.key private.key
```

## Certificate Sources

### Let's Encrypt (Recommended for production)
```bash
# Install certbot
sudo apt-get install certbot

# Generate certificate
sudo certbot certonly --standalone -d yourdomain.com

# Copy to this directory
sudo cp /etc/letsencrypt/live/yourdomain.com/fullchain.pem .
sudo cp /etc/letsencrypt/live/yourdomain.com/privkey.pem ./private.key
```

### Self-Signed Certificate (Development only)
```bash
# Generate self-signed certificate
openssl req -x509 -newkey rsa:4096 -keyout private.key -out cert.pem -days 365 -nodes

# Create full chain (same as cert for self-signed)
cp cert.pem fullchain.pem
```

## Environment Configuration

Update your environment files to reference these certificates:

```env
# In production.env
SSL_CERT_PATH=/app/config/ssl/fullchain.pem
SSL_KEY_PATH=/app/config/ssl/private.key
ENABLE_SSL=true
```

## Docker Configuration

The Docker setup automatically mounts this directory:

```yaml
volumes:
  - ./config/ssl:/app/config/ssl:ro
```

## Security Notes

⚠️ **IMPORTANT**: Never commit certificate files or private keys to version control!

- Certificate files are automatically ignored by `.gitignore`
- Use environment variables to reference certificate paths
- Regularly renew certificates (Let's Encrypt certificates expire every 90 days)
- Monitor certificate expiration dates

## Nginx Configuration

The certificates are referenced in the Nginx configuration:

```nginx
ssl_certificate /etc/nginx/ssl/fullchain.pem;
ssl_certificate_key /etc/nginx/ssl/private.key;
```
