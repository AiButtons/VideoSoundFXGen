#!/bin/bash

# Check if password argument is provided
if [ $# -ne 1 ]; then
    echo "Usage: $0 <password>"
    exit 1
fi

PASSWORD=$1

# Initialize endpoints file
echo '{}' > endpoints.json

# Create Modal secret with provided password
modal secret create api-key api_key="$PASSWORD"

# Function to extract endpoint URL
extract_endpoint() {
    grep -o 'https://.*modal\.run' | head -n 1
}

# List of models to deploy
models=("fasthunyuan.py" "mmaudio.py")

for model in "${models[@]}"; do
    model_name=$(basename "$model" .py)
    echo "Deploying $model_name..."
    
    endpoint=$(modal deploy "$model" 2>&1 | extract_endpoint)
    
    if [ -n "$endpoint" ]; then
        content=$(cat endpoints.json)
        content=${content%\}}
        if [ "$content" = "{" ]; then
            echo "$content\"$model_name\": \"$endpoint\"}" > endpoints.json
        else
            echo "$content, \"$model_name\": \"$endpoint\"}" > endpoints.json
        fi
        echo "✓ Stored endpoint for $model_name: $endpoint"
    else
        echo "⨯ Failed to get endpoint for $model_name"
    fi
done