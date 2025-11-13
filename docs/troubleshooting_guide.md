# RAG Education Assistant - Troubleshooting Guide

## Overview

This guide provides solutions to common issues encountered when deploying, configuring, and operating the RAG Education Assistant system.

## Quick Diagnostic Commands

### System Health Check

```bash
# Run comprehensive system validation
python scripts/run_system_validation.py

# Check service status
docker-compose ps

# Check service logs
docker-compose logs --tail=50 -f
```

### Service-Specific Health

```bash
# Auth Service
curl http://localhost:8006/health

# API Gateway
curl http://localhost:8000/health/comprehensive

# Frontend
curl http://localhost:3000
```

## Authentication Issues

### Issue: "Invalid credentials" on login

**Symptoms:**

- Users cannot login with correct credentials
- 401 Unauthorized errors
- Login form shows "Invalid username or password"

**Possible Causes:**

1. Database not initialized
2. Demo users not created
3. Password hash mismatch
4. Auth service not running

**Solutions:**

```bash
# 1. Check if auth service is running
curl http://localhost:8006/health

# 2. Verify database exists and has data
sqlite3 data/rag_assistant.db ".tables"
sqlite3 data/rag_assistant.db "SELECT username, role_id FROM users;"

# 3. Recreate database and demo users
python src/database/create_admin.py

# 4. Reset admin password
python src/database/create_admin.py --reset-admin-password

# 5. Check auth service logs
docker-compose logs auth-service
```

### Issue: "JWT token expired" errors

**Symptoms:**

- Users get logged out frequently
- API calls return 401 after some time
- Token refresh not working

**Possible Causes:**

1. Token expiration time too short
2. Refresh token mechanism broken
3. System time synchronization issues

**Solutions:**

```bash
# 1. Check token configuration
grep TOKEN_EXPIRE .env.development

# 2. Increase token expiration (development only)
echo "ACCESS_TOKEN_EXPIRE_MINUTES=60" >> .env.development

# 3. Check system time
date
timedatectl status

# 4. Test token refresh endpoint
curl -X POST http://localhost:8006/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"your-refresh-token"}'
```

### Issue: "CORS policy" errors in browser

**Symptoms:**

- Frontend cannot connect to backend
- Browser console shows CORS errors
- Network requests blocked by browser

**Solutions:**

```bash
# 1. Check CORS configuration in auth service
grep CORS_ORIGINS .env.development

# 2. Update CORS origins to include frontend URL
echo "CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000" >> .env.development

# 3. Restart auth service
docker-compose restart auth-service

# 4. Clear browser cache and cookies
# 5. Check browser developer tools Network tab
```

## Database Issues

### Issue: "Database locked" error

**Symptoms:**

- SQLite database locked errors
- Services fail to start
- "database is locked" in logs

**Solutions:**

```bash
# 1. Find processes using the database
lsof data/rag_assistant.db

# 2. Kill processes if necessary
pkill -f rag_assistant.db

# 3. Enable WAL mode for better concurrency
sqlite3 data/rag_assistant.db "PRAGMA journal_mode=WAL;"

# 4. Check database integrity
sqlite3 data/rag_assistant.db "PRAGMA integrity_check;"

# 5. Restart services
docker-compose restart
```

### Issue: "No such table: users" error

**Symptoms:**

- Database errors mentioning missing tables
- Services fail to start with table errors
- Fresh installation fails

**Solutions:**

```bash
# 1. Check if database file exists
ls -la data/rag_assistant.db

# 2. Check database schema
sqlite3 data/rag_assistant.db ".schema"

# 3. Recreate database
rm data/rag_assistant.db
python src/database/create_admin.py

# 4. Verify tables created
sqlite3 data/rag_assistant.db ".tables"
```

### Issue: Foreign key constraint errors

**Symptoms:**

- Cannot create users or sessions
- Constraint violation errors in logs
- Database operations fail

**Solutions:**

```bash
# 1. Check foreign key constraints
sqlite3 data/rag_assistant.db "PRAGMA foreign_key_check;"

# 2. Verify referential integrity
sqlite3 data/rag_assistant.db "
SELECT u.username, u.role_id, r.name
FROM users u
LEFT JOIN roles r ON u.role_id = r.role_id
WHERE r.role_id IS NULL;"

# 3. Fix orphaned records
sqlite3 data/rag_assistant.db "
DELETE FROM users WHERE role_id NOT IN (SELECT role_id FROM roles);"

# 4. Recreate default roles if missing
python src/database/create_admin.py --create-roles-only
```

## Service Connection Issues

### Issue: Services cannot connect to each other

**Symptoms:**

- Connection refused errors between services
- API Gateway cannot reach Auth Service
- Health checks failing

**Solutions:**

```bash
# 1. Check service URLs in environment
grep SERVICE_URL .env.development

# 2. Verify services are running and ports are correct
docker-compose ps
netstat -tlnp | grep -E "(8000|8006|3000)"

# 3. Test connectivity between containers
docker-compose exec api-gateway curl http://auth-service:8006/health

# 4. Check Docker network
docker network ls
docker network inspect rag3_for_colab_default

# 5. Restart with fresh network
docker-compose down
docker-compose up -d
```

### Issue: "Service unavailable" errors

**Symptoms:**

- External microservices not responding
- PDF processing fails
- Model inference errors

**Solutions:**

```bash
# 1. Check external service URLs
curl https://pdf-processor-awe3elsvra-ew.a.run.app/health
curl https://model-inferencer-awe3elsvra-ew.a.run.app/health

# 2. Check API keys if required
grep API_KEY .env.development

# 3. Test service endpoints directly
curl -X POST https://pdf-processor-awe3elsvra-ew.a.run.app/convert/pdf-to-markdown \
  -F "file=@test.pdf"

# 4. Check service logs for specific errors
docker-compose logs api-gateway | grep -i "service"
```

## Frontend Issues

### Issue: Frontend won't start or shows errors

**Symptoms:**

- Next.js build failures
- White screen on frontend
- JavaScript errors in browser

**Solutions:**

```bash
# 1. Check Node.js version
node --version  # Should be 18+

# 2. Clear npm cache and reinstall
cd frontend
rm -rf node_modules package-lock.json
npm cache clean --force
npm install

# 3. Check environment variables
cat .env.local

# 4. Run development server with debug
npm run dev

# 5. Check browser developer tools console
```

### Issue: Authentication not working in frontend

**Symptoms:**

- Login form doesn't respond
- Auth context not working
- User state not persisting

**Solutions:**

```bash
# 1. Check frontend environment variables
cat frontend/.env.local

# 2. Verify API URLs are correct
grep NEXT_PUBLIC_API_URL frontend/.env.local

# 3. Check AuthContext implementation
cat frontend/contexts/AuthContext.tsx | grep -A 10 "const login"

# 4. Clear browser storage
# Open browser dev tools → Application → Storage → Clear All

# 5. Check network requests in dev tools
# Dev Tools → Network → XHR/Fetch filter
```

## Docker Issues

### Issue: Docker containers won't start

**Symptoms:**

- `docker-compose up` fails
- Container exit codes
- Build failures

**Solutions:**

```bash
# 1. Check Docker system status
docker system info
docker system df

# 2. Clean Docker system
docker system prune -a

# 3. Check logs for specific container
docker-compose logs auth-service

# 4. Build without cache
docker-compose build --no-cache

# 5. Check Dockerfile syntax
docker build -t test-build -f services/auth_service/Dockerfile .
```

### Issue: Port conflicts

**Symptoms:**

- "Port already in use" errors
- Services fail to bind to ports
- Multiple instances running

**Solutions:**

```bash
# 1. Check what's using the ports
sudo lsof -i :8006
sudo lsof -i :8000
sudo lsof -i :3000

# 2. Kill processes using ports
sudo kill -9 <PID>

# 3. Use different ports in docker-compose.yml
# Change port mappings: "8007:8006" instead of "8006:8006"

# 4. Stop all containers and restart
docker-compose down --remove-orphans
docker-compose up -d
```

## Performance Issues

### Issue: Slow response times

**Symptoms:**

- API requests taking too long
- Database queries slow
- High CPU/memory usage

**Solutions:**

```bash
# 1. Check system resources
htop
df -h
free -h

# 2. Check database performance
sqlite3 data/rag_assistant.db "
.timer on
SELECT COUNT(*) FROM users;
.timer off"

# 3. Analyze database
sqlite3 data/rag_assistant.db "ANALYZE;"
sqlite3 data/rag_assistant.db "PRAGMA optimize;"

# 4. Check Docker resources
docker stats

# 5. Add database indexes if needed
sqlite3 data/rag_assistant.db "
CREATE INDEX IF NOT EXISTS idx_users_last_login ON users(last_login);
CREATE INDEX IF NOT EXISTS idx_sessions_last_accessed ON user_sessions(last_accessed);"
```

### Issue: Memory leaks

**Symptoms:**

- Memory usage constantly increasing
- Services become unresponsive
- Out of memory errors

**Solutions:**

```bash
# 1. Monitor memory usage over time
watch -n 5 'docker stats --no-stream'

# 2. Check for memory leaks in logs
docker-compose logs | grep -i "memory\|leak"

# 3. Restart services periodically (temporary fix)
docker-compose restart

# 4. Increase memory limits in docker-compose.yml
# Add to service configuration:
# mem_limit: 512m
# mem_reservation: 256m

# 5. Update to latest container images
docker-compose pull
docker-compose up -d
```

## File and Permission Issues

### Issue: File permission errors

**Symptoms:**

- Cannot read/write database file
- File upload errors
- Permission denied errors

**Solutions:**

```bash
# 1. Check file permissions
ls -la data/
ls -la data/rag_assistant.db

# 2. Fix permissions
sudo chown -R $USER:$USER data/
chmod 644 data/rag_assistant.db
chmod 755 data/

# 3. Check Docker volume permissions
docker-compose exec auth-service ls -la /data/

# 4. Fix Docker permissions
docker-compose exec auth-service chown -R app:app /data/
```

### Issue: File upload failures

**Symptoms:**

- Document upload returns errors
- PDF conversion fails
- File size limit errors

**Solutions:**

```bash
# 1. Check file size limits
grep client_max_body_size /etc/nginx/sites-enabled/rag-assistant

# 2. Check available disk space
df -h

# 3. Check file permissions in upload directory
ls -la data/markdown/

# 4. Test file upload directly
curl -F "file=@test.pdf" http://localhost:8000/documents/convert-document-to-markdown

# 5. Check API Gateway logs
docker-compose logs api-gateway | grep -i upload
```

## Security Issues

### Issue: SSL/TLS certificate problems

**Symptoms:**

- HTTPS connections fail
- Certificate warnings in browser
- SSL handshake errors

**Solutions:**

```bash
# 1. Check certificate validity
openssl x509 -in /etc/ssl/certs/yourdomain.com.crt -text -noout

# 2. Test SSL connection
openssl s_client -connect yourdomain.com:443 -servername yourdomain.com

# 3. Check certificate chain
curl -vI https://yourdomain.com

# 4. Renew Let's Encrypt certificate
sudo certbot renew --nginx

# 5. Test nginx configuration
sudo nginx -t
```

### Issue: Rate limiting too aggressive

**Symptoms:**

- Users getting rate limited quickly
- 429 Too Many Requests errors
- Normal usage blocked

**Solutions:**

```bash
# 1. Check rate limiting configuration
grep RATE_LIMIT .env.development

# 2. Increase rate limits temporarily
echo "RATE_LIMIT_RPM=120" >> .env.development
echo "RATE_LIMIT_BURST=30" >> .env.development

# 3. Restart auth service
docker-compose restart auth-service

# 4. Monitor rate limiting
docker-compose logs auth-service | grep -i "rate"

# 5. Implement IP whitelisting for admin users
```

## Integration Testing Issues

### Issue: Integration tests failing

**Symptoms:**

- Test suite returns errors
- Services not ready for tests
- Database state issues

**Solutions:**

```bash
# 1. Run tests with detailed output
python scripts/run_system_validation.py -v

# 2. Check test database
ls -la tests/test_*.db

# 3. Clean test environment
rm -rf tests/__pycache__/
rm -f tests/test_*.db

# 4. Run tests individually
cd tests/integration
python -m pytest test_auth_system.py -v

# 5. Check test dependencies
pip install -r tests/requirements.txt
```

## Log Analysis

### Useful Log Commands

```bash
# Search for errors across all services
docker-compose logs | grep -i error

# Monitor real-time logs
docker-compose logs -f --tail=100

# Search for authentication issues
docker-compose logs auth-service | grep -E "(401|403|login|token)"

# Check database operations
docker-compose logs | grep -i "database\|sqlite"

# Monitor API requests
docker-compose logs api-gateway | grep -E "(POST|GET|PUT|DELETE)"

# Export logs for analysis
docker-compose logs > system_logs_$(date +%Y%m%d_%H%M).log
```

### Log Locations

**Development:**

- Docker logs: `docker-compose logs [service]`
- Frontend logs: Browser dev tools console
- Build logs: Terminal output during `docker-compose up`

**Production:**

- System logs: `/var/log/rag-assistant/`
- Nginx logs: `/var/log/nginx/access.log` and `/var/log/nginx/error.log`
- Application logs: `docker-compose logs` or configured log drivers

## Monitoring and Alerting

### Set Up Basic Monitoring

```bash
# Create monitoring script
cat > monitor.sh << 'EOF'
#!/bin/bash
# Basic health monitoring

SERVICES=("http://localhost:8006/health" "http://localhost:8000/health" "http://localhost:3000")
EMAILS="admin@yourdomain.com"

for service in "${SERVICES[@]}"; do
    if ! curl -f -s "$service" > /dev/null; then
        echo "Service $service is down at $(date)" | mail -s "RAG Assistant Alert" "$EMAILS"
    fi
done
EOF

chmod +x monitor.sh

# Add to crontab (check every 5 minutes)
(crontab -l 2>/dev/null; echo "*/5 * * * * /path/to/monitor.sh") | crontab -
```

### Performance Monitoring

```bash
# Create performance monitoring
cat > perf_monitor.sh << 'EOF'
#!/bin/bash
# Performance monitoring

LOG_FILE="/var/log/rag-performance.log"

# System metrics
echo "$(date): CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)" >> $LOG_FILE
echo "$(date): Memory: $(free | grep Mem | awk '{printf "%.2f", ($3/$2)*100}')" >> $LOG_FILE
echo "$(date): Disk: $(df -h / | awk 'NR==2{print $5}' | cut -d'%' -f1)" >> $LOG_FILE

# Service response times
for url in "http://localhost:8006/health" "http://localhost:8000/health"; do
    response_time=$(curl -w "%{time_total}" -o /dev/null -s "$url")
    echo "$(date): $url response time: ${response_time}s" >> $LOG_FILE
done
EOF

chmod +x perf_monitor.sh
```

## Recovery Procedures

### Complete System Recovery

```bash
#!/bin/bash
# Emergency recovery procedure

echo "Starting emergency recovery..."

# 1. Stop all services
docker-compose down

# 2. Backup current state
mkdir -p recovery_$(date +%Y%m%d_%H%M)
cp -r data/ recovery_$(date +%Y%m%d_%H%M)/
cp .env.* recovery_$(date +%Y%m%d_%H%M)/

# 3. Clean Docker environment
docker system prune -af --volumes

# 4. Restore from last known good backup
if [ -f "data/backup_latest.db" ]; then
    cp data/backup_latest.db data/rag_assistant.db
else
    echo "No backup found, reinitializing database..."
    python src/database/create_admin.py
fi

# 5. Rebuild and start services
docker-compose build --no-cache
docker-compose up -d

# 6. Wait for services to be ready
sleep 60

# 7. Validate recovery
python scripts/run_system_validation.py

echo "Recovery procedure completed"
```

### Data Recovery

```bash
# Recover from database corruption
sqlite3 data/rag_assistant.db ".recover" > recovered_data.sql
rm data/rag_assistant.db
sqlite3 data/rag_assistant.db < recovered_data.sql

# Verify recovery
sqlite3 data/rag_assistant.db "PRAGMA integrity_check;"
```

## Getting Help

### Information to Collect

When seeking help, collect this information:

```bash
# System information
python scripts/run_system_validation.py --save-report debug_report.json

# Service versions
docker-compose version
python --version
node --version

# System logs
docker-compose logs > debug_logs.txt

# Environment configuration (sanitize secrets first!)
env | grep -E "(JWT|API|SERVICE)" > debug_env.txt
```

### Support Channels

1. **Documentation**: Check all files in `docs/` directory
2. **Issue Tracker**: Create detailed bug reports with debug information
3. **Community Forums**: Search for similar issues and solutions
4. **Professional Support**: Contact system administrators with debug package

### Creating Debug Package

```bash
#!/bin/bash
# Create comprehensive debug package

DEBUG_DIR="debug_package_$(date +%Y%m%d_%H%M)"
mkdir -p "$DEBUG_DIR"

# System validation
python scripts/run_system_validation.py --save-report "$DEBUG_DIR/validation.json" --skip-integration

# Service logs
docker-compose logs > "$DEBUG_DIR/service_logs.txt"

# Configuration (sanitized)
cp .env.example "$DEBUG_DIR/"
env | grep -v -E "(SECRET|KEY|PASSWORD)" > "$DEBUG_DIR/environment.txt"

# System info
docker --version > "$DEBUG_DIR/system_info.txt"
docker-compose --version >> "$DEBUG_DIR/system_info.txt"
python --version >> "$DEBUG_DIR/system_info.txt"

# Database info (if accessible)
if [ -f "data/rag_assistant.db" ]; then
    sqlite3 data/rag_assistant.db ".tables" > "$DEBUG_DIR/database_tables.txt"
    sqlite3 data/rag_assistant.db "SELECT COUNT(*) as user_count FROM users;" >> "$DEBUG_DIR/database_info.txt" 2>/dev/null
fi

tar -czf "${DEBUG_DIR}.tar.gz" "$DEBUG_DIR"
echo "Debug package created: ${DEBUG_DIR}.tar.gz"
```

This troubleshooting guide should help resolve most common issues encountered with the RAG Education Assistant system.
