# Google Cloud Run Deployment Script for Research Router Web App (PowerShell)
# Usage: .\deploy.ps1 [service-name] [region]

param(
    [string]$ServiceName = "research-router",
    [string]$Region = "us-central1"
)

Write-Host "🚀 Deploying Research Router to Google Cloud Run..." -ForegroundColor Green
Write-Host ""

# Get project ID
$ProjectId = gcloud config get-value project
if (-not $ProjectId) {
    Write-Host "❌ Error: No project set" -ForegroundColor Red
    Write-Host "Run: gcloud config set project YOUR_PROJECT_ID" -ForegroundColor Yellow
    exit 1
}

Write-Host "Project: $ProjectId" -ForegroundColor Cyan
Write-Host "Service: $ServiceName" -ForegroundColor Cyan
Write-Host "Region: $Region" -ForegroundColor Cyan
Write-Host ""

# Check if gcloud is authenticated
$AuthCheck = gcloud auth list --filter=status:ACTIVE --format="value(account)" 2>$null
if (-not $AuthCheck) {
    Write-Host "❌ Error: Not authenticated with Google Cloud" -ForegroundColor Red
    Write-Host "Run: gcloud auth login" -ForegroundColor Yellow
    exit 1
}

# Enable required APIs
Write-Host "📋 Enabling required Google Cloud APIs..." -ForegroundColor Blue
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Check for required environment variables
Write-Host "🔍 Checking for required environment variables..." -ForegroundColor Blue

if (-not $env:OPENAI_API_KEY) {
    Write-Host "❌ Error: OPENAI_API_KEY environment variable not set" -ForegroundColor Red
    Write-Host "Set it with: `$env:OPENAI_API_KEY=`"your_key_here`"" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

if (-not $env:SECRET_KEY) {
    Write-Host "⚠️  Warning: SECRET_KEY not set, generating random key..." -ForegroundColor Yellow
    $env:SECRET_KEY = python -c "import secrets; print(secrets.token_urlsafe(32))"
}

# Build and deploy to Cloud Run
Write-Host "🏗️  Building and deploying to Cloud Run..." -ForegroundColor Blue

$deployCmd = @(
    "gcloud", "run", "deploy", $ServiceName,
    "--source", ".",
    "--platform", "managed",
    "--region", $Region,
    "--allow-unauthenticated",
    "--memory", "1Gi",
    "--cpu", "1",
    "--timeout", "300",
    "--set-env-vars", "OPENAI_API_KEY=$env:OPENAI_API_KEY,SECRET_KEY=$env:SECRET_KEY,FLASK_ENV=production"
)

& $deployCmd[0] $deployCmd[1..$deployCmd.Length]

if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Deployment failed!" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Get the service URL
$ServiceUrl = gcloud run services describe $ServiceName --region $Region --format="value(status.url)"

Write-Host ""
Write-Host "✅ Deployment successful!" -ForegroundColor Green
Write-Host "🌐 Service URL: $ServiceUrl" -ForegroundColor Cyan
Write-Host ""
Write-Host "📋 Next steps:" -ForegroundColor Yellow
Write-Host "1. Test your app: curl $ServiceUrl/health"
Write-Host "2. To map custom domain: gcloud run domain-mappings create --service $ServiceName --domain yyy.xxx.com --region $Region"
Write-Host "3. Update your DNS: Add CNAME record 'yyy' pointing to 'ghs.googlehosted.com'"
Write-Host ""
Write-Host "💰 Estimated cost: `$0-10/month (pay per request, scales to zero)" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit"