# IndoorAI — Production Deployment & Setup Guide

**Version**: 1.0.0  
**Target Systems**: macOS, Linux (Ubuntu 22.04 LTS), Docker, Android 10+  

---

## 1. System Requirements

### Backend & Core Algorithms
- **Python**: 3.10 or 3.11 (Python 3.11 recommended).
- **RAM**: Minimum 4 GB (8 GB recommended for Deep Learning training and multi-session concurrency).
- **Disk**: 5 GB free disk space (for models, datasets, and cache).
- **Operating Systems**: macOS (Apple Silicon / Intel), Linux (Debian/Ubuntu/CentOS).

### Android Client (Build Machine)
- **Java Development Kit**: JDK 17 or JDK 21.
- **Android SDK**: API Level 34 (Android 14) platform tools and build tools.
- **Gradle**: Gradle 8.2+ (bundled via `./gradlew`).

---

## 2. Environment Variables & Configuration

Create a `.env` file in the repository root by copying `.env.example`:

```bash
cp .env.example .env
```

### Essential Configuration Keys

```ini
# ==============================================================================
# Security & Authentication (RFC 7519 JWT)
# ==============================================================================
# Secret key used for signing HMAC-SHA256 tokens.
# CRITICAL: Change to a high-entropy 256-bit secret key in production!
JWT_SECRET_KEY=replace-with-a-secure-random-64-character-hex-string-for-production

# Cryptographic algorithm (HS256 supported out-of-the-box)
JWT_ALGORITHM=HS256

# Token expiration lifetime in minutes (default: 60)
ACCESS_TOKEN_EXPIRE_MINUTES=60

# Strict Authentication Mode:
# Set to 'true' to reject all unauthenticated requests with HTTP 401.
# Set to 'false' in local dev mode to allow unauthenticated requests with guest claims.
INDOORAI_STRICT_AUTH=true

# ==============================================================================
# Server & Network Configuration
# ==============================================================================
HOST=0.0.0.0
PORT=8000
ENVIRONMENT=production
DEBUG=false
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8501

# ==============================================================================
# Storage & Directories
# ==============================================================================
DATASET_STORAGE_PATH=./data/raw
PROCESSED_DATA_PATH=./data/processed
MODELS_PATH=./experiments
RADIO_MAPS_PATH=./experiments/wifi_localization/radio_maps
MAPS_STORAGE_PATH=./data/maps
```

---

## 3. Backend Installation & Setup

### Step 1: Clone Repository & Setup Virtual Environment
```bash
git clone https://github.com/bineetkeshari/IndoorAI.git
cd IndoorAI

python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Run Validation & Test Suite
Verify that all 323 automated unit and integration tests pass:
```bash
python -m unittest discover -s tests -p "test_*.py"
```

### Step 4: Run Performance Benchmarks
Run the system benchmark harness to verify SLA compliance on your host:
```bash
python src/evaluation/system_benchmark.py
```

---

## 4. Running the Backend Services

### Starting the FastAPI Server (Production Uvicorn)
```bash
uvicorn backend.app.main:app \
  --host 0.0.0.0 \
  --port 8000 \
  --workers 4 \
  --log-level info
```

For development with live reloading:
```bash
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Interactive API Documentation
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

---

## 5. Starting the Interactive Streamlit Dashboard

IndoorAI includes a full-featured Streamlit visualization dashboard for dataset exploration, model benchmarks, radio-map inspection, and navigation replay:

```bash
streamlit run src/dashboard/app.py --server.port 8501 --server.address 0.0.0.0
```

Access the dashboard at `http://localhost:8501`.

---

## 6. Android Mobile Application Build & Deployment

### Step 1: Set Android SDK Location
Ensure `local.properties` in `android/` contains your SDK directory:
```properties
sdk.dir=/Users/username/Library/Android/sdk
```

### Step 2: Configure Server URL
In `android/app/src/main/res/values/strings.xml`, update the default server IP:
```xml
<string name="default_server_url">ws://192.168.1.100:8000/api/realtime/sessions/</string>
```

### Step 3: Build Debug APK
```bash
cd android
./gradlew assembleDebug
```
The output APK is generated at:
`android/app/build/outputs/apk/debug/app-debug.apk`

### Step 4: Install to Connected Device
```bash
adb install -r app/build/outputs/apk/debug/app-debug.apk
```

---

## 7. Production Deployment & Security Hardening

### 7.1 Reverse Proxy Configuration (Nginx)
Use Nginx as an SSL/TLS termination proxy for secure HTTPS and WSS (WebSocket Secure):

```nginx
server {
    listen 443 ssl http2;
    server_name indoorai.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/indoorai.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/indoorai.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_read_timeout 86400s;
        proxy_send_timeout 86400s;
    }
}
```

### 7.2 Security Checklist
- [x] Set a unique, high-entropy `JWT_SECRET_KEY` in environment secrets (never commit to git).
- [x] Enforce `INDOORAI_STRICT_AUTH=true` in production to prevent unauthenticated access.
- [x] Terminate TLS at the reverse proxy to encrypt all Wi-Fi, IMU, and location streams in transit.
- [x] Configure CORS (`ALLOWED_ORIGINS`) to only trust authorized client domains.
- [x] Ensure on-device face blurring and EXIF sanitization are enabled before storing photos.

---

## 8. Smoke Test & Health Check

Run a quick health check against the live server:

```bash
# 1. Health Probe
curl -s http://localhost:8000/health | jq .

# 2. Issue a Test JWT Token
DEV_TOKEN=$(curl -s -X POST http://localhost:8000/api/auth/dev-token \
  -H "Content-Type: application/json" \
  -d '{"user_id": "SURVEYOR-TEST", "role": "ADMIN"}' | jq -r .access_token)

# 3. Verify Token
curl -s http://localhost:8000/api/auth/verify \
  -H "Authorization: Bearer $DEV_TOKEN" | jq .

# 4. Check Real-Time Streaming Status
curl -s http://localhost:8000/api/realtime/sessions \
  -H "Authorization: Bearer $DEV_TOKEN" | jq .
```
