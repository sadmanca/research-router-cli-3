#!/bin/bash

# Google Cloud Run Deployment Script for Research Router Web App
# Usage: ./deploy.sh [service-name] [region]

set -e

# Configuration
SERVICE_NAME=${1:-"research-router"}
REGION=${2:-"us-central1"}
PROJECT_ID=$(gcloud config get-value project)

echo "🚀 Deploying Research Router to Google Cloud Run..."
echo "Project: $PROJECT_ID"
echo "Service: $SERVICE_NAME"
echo "Region: $REGION"
echo ""

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q .; then
    echo "❌ Error: Not authenticated with Google Cloud"
    echo "Run: gcloud auth login"
    exit 1
fi

# Check if project is set
if [ -z "$PROJECT_ID" ]; then
    echo "❌ Error: No project set"
    echo "Run: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Enable required APIs
echo "📋 Enabling required Google Cloud APIs..."
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com

# Check for required environment variables
echo "🔍 Checking for required environment variables..."
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY environment variable not set"
    echo "Set it with: export OPENAI_API_KEY=your_key_here"
    exit 1
fi

if [ -z "$SECRET_KEY" ]; then
    echo "⚠️  Warning: SECRET_KEY not set, generating random key..."
    SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_urlsafe(32))")
fi

# Build and deploy to Cloud Run
echo "🏗️  Building and deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --source . \
    --platform managed \
    --region $REGION \
    --allow-unauthenticated \
    --memory 1Gi \
    --cpu 1 \
    --timeout 300 \
    --set-env-vars "OPENAI_API_KEY=$OPENAI_API_KEY,SECRET_KEY=$SECRET_KEY,FLASK_ENV=production"

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --format="value(status.url)")

echo ""
echo "✅ Deployment successful!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "📋 Next steps:"
echo "1. Test your app: curl $SERVICE_URL/health"
echo "2. To map custom domain: gcloud run domain-mappings create --service $SERVICE_NAME --domain yyy.xxx.com --region $REGION"
echo "3. Update your DNS: Add CNAME record 'yyy' pointing to 'ghs.googlehosted.com'"
echo ""
echo "💰 Estimated cost: \$0-10/month (pay per request, scales to zero)"