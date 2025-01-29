#!/bin/bash

# Check if password argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <password>"
    exit 1
fi

# Your password
PASSWORD=$1

# Colors for better log visibility
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to extract endpoint URL
extract_endpoint() {
    grep -o 'https://.*modal\.run' | head -n 1
}

# Function to deploy Modal models
deploy_modal() {
    # Navigate to modal directory
    if ! cd modal; then
        echo -e "${RED}Error: Could not find modal directory${NC}"
        exit 1
    fi

    # Initialize endpoints file
    echo '{}' > endpoints.json

    # Create Modal secret with provided password
    echo -e "${BLUE}Creating Modal secret...${NC}"
    modal secret create api-key api_key=$PASSWORD

    # List of models to deploy
    models=("fasthunyuan.py" "mmaudio.py")

    for model in "${models[@]}"; do
        model_name=$(basename "$model" .py)
        echo -e "${BLUE}Deploying $model_name...${NC}"
        
        endpoint=$(modal deploy "$model" 2>&1 | tee /dev/tty | extract_endpoint)
        
        if [ -n "$endpoint" ]; then
            content=$(cat endpoints.json)
            content=${content%\}}
            if [ "$content" = "{" ]; then
                echo "$content\"$model_name\": \"$endpoint\"}" > endpoints.json
            else
                echo "$content, \"$model_name\": \"$endpoint\"}" > endpoints.json
            fi
            echo -e "${GREEN}✓ Stored endpoint for $model_name: $endpoint${NC}"
        else
            echo -e "${RED}⨯ Failed to get endpoint for $model_name${NC}"
            exit 1
        fi
    done

    # Return to original directory
    cd ..
}

# Main execution
main() {
    # Store the original directory
    ORIGINAL_DIR=$(pwd)

    # Deploy Modal first
    deploy_modal

    # Navigate to cdk directory
    cd "$ORIGINAL_DIR/cdk" || {
        echo -e "${RED}Error: Could not find cdk directory${NC}"
        exit 1
    }

    # Deploy CDK stack
    echo -e "\n${BLUE}====== CDK Deployment Logs Start ======${NC}"
    if LAMBDA_URL=$(cdk deploy --parameters ApiKey="$PASSWORD" 2>&1 | tee /dev/tty | grep "FastAPIFunctionUrlOutput = " | cut -d "=" -f2- | tr -d ' '); then
        echo -e "${GREEN}✓ CDK deployment completed successfully${NC}"
        echo -e "\n${GREEN}🚀 Check out your deployed app at:${NC} ${BLUE}${LAMBDA_URL}${NC}\n"
    else
        echo -e "${RED}⨯ CDK deployment failed${NC}"
        exit 1
    fi
    echo -e "${BLUE}====== CDK Deployment Logs End ======${NC}"
}

# Execute main function
main