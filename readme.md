# AI Video & Sound Generator

A Gradio app deployed on AWS Lambda using web adapter that lets you generate videos from text and add AI-generated sound effects. Uses FastHunyuan for video and MMAudio for audio generation.

## Models
- Video: [FastHunyuan](https://huggingface.co/FastVideo/FastHunyuan)
- Audio: [MMAudio](https://github.com/hkchengrex/MMAudio)

## Requirements
- Modal account
- AWS account
- Node.js 18+ (for CDK)
- WSL/ Linux/ macOS (bash shell required)

## Deploy

1. Modal setup:
```bash
pip install modal
python -m modal setup
```

2. AWS & CDK setup:
```bash
aws configure
npm install -g aws-cdk
cdk bootstrap
```

3. Run deployment:
```bash
chmod +x setup.sh
./setup.sh <your-password>  # Password used for endpoint authentication
```

The script:
- Deploys models to Modal with authentication
- Creates JWT-secured endpoints using provided password
- Deploys Gradio interface on Lambda with web adapter, and provides the lambda url

## Authentication
The password provided to setup.sh is used to:
- Create secure Modal endpoints
- Generate JWTs for API authentication
- Secure communication between Lambda and Modal

## Access
Open the Lambda URL provided after deployment to use the Gradio interface.
