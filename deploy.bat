@echo off
REM Google Cloud Run Deployment Script for Research Router Web App
REM Usage: deploy.bat [service-name] [region]

setlocal enabledelayedexpansion

REM Configuration
set SERVICE_NAME=%1
if "%SERVICE_NAME%"=="" set SERVICE_NAME=research-router

set REGION=%2
if "%REGION%"=="" set REGION=us-central1

echo 🚀 Deploying Research Router to Google Cloud Run...

REM Get project ID
for /f "tokens=*" %%i in ('gcloud config get-value project') do set PROJECT_ID=%%i

echo Project: %PROJECT_ID%
echo Service: %SERVICE_NAME%
echo Region: %REGION%
echo.

REM Check if gcloud is authenticated
gcloud auth list --filter=status:ACTIVE --format="value(account)" >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Not authenticated with Google Cloud
    echo Run: gcloud auth login
    exit /b 1
)

REM Check if project is set
if "%PROJECT_ID%"=="" (
    echo ❌ Error: No project set
    echo Run: gcloud config set project YOUR_PROJECT_ID
    exit /b 1
)

REM Enable required APIs
echo 📋 Enabling required Google Cloud APIs...
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

REM Check for required environment variables
echo 🔍 Checking for required environment variables...
if "%OPENAI_API_KEY%"=="" (
    echo ❌ Error: OPENAI_API_KEY environment variable not set
    echo Set it with: set OPENAI_API_KEY=your_key_here
    echo Or in PowerShell: $env:OPENAI_API_KEY="your_key_here"
    pause
    exit /b 1
)

if "%SECRET_KEY%"=="" (
    echo ⚠️  Warning: SECRET_KEY not set, generating random key...
    for /f "tokens=*" %%i in ('python -c "import secrets; print(secrets.token_urlsafe(32))"') do set SECRET_KEY=%%i
)

REM Build and deploy to Cloud Run
echo 🏗️  Building and deploying to Cloud Run...
gcloud run deploy %SERVICE_NAME% ^
    --source . ^
    --platform managed ^
    --region %REGION% ^
    --allow-unauthenticated ^
    --memory 1Gi ^
    --cpu 1 ^
    --timeout 300 ^
    --set-env-vars "OPENAI_API_KEY=%OPENAI_API_KEY%,SECRET_KEY=%SECRET_KEY%,FLASK_ENV=production"

if errorlevel 1 (
    echo ❌ Deployment failed!
    pause
    exit /b 1
)

REM Get the service URL
for /f "tokens=*" %%i in ('gcloud run services describe %SERVICE_NAME% --region %REGION% --format="value(status.url)"') do set SERVICE_URL=%%i

echo.
echo ✅ Deployment successful!
echo 🌐 Service URL: %SERVICE_URL%
echo.
echo 📋 Next steps:
echo 1. Test your app: curl %SERVICE_URL%/health
echo 2. To map custom domain: gcloud run domain-mappings create --service %SERVICE_NAME% --domain yyy.xxx.com --region %REGION%
echo 3. Update your DNS: Add CNAME record 'yyy' pointing to 'ghs.googlehosted.com'
echo.
echo 💰 Estimated cost: $0-10/month (pay per request, scales to zero)
echo.
pause