# flow, first cd to modal and create secret and dpeloy the models, also add that secret to cdk.json, along with modal endpoints, then deploy it,

#config parser

#!/bin/bash

# Your password
PASSWORD="your_secret_password"

# Navigate to modal directory and check if successful
if ! cd modal; then
    echo "Error: Could not find modal directory"
    exit 1
fi

# Check if deploy_modal.sh exists and make it executable
if [ -f deploy_modal.sh ]; then
    chmod +x deploy_modal.sh
else
    echo "Error: deploy_modal.sh not found"
    exit 1
fi

# Run deploy script with password
./deploy_modal.sh "$PASSWORD"

# then merge it with cdk.json, then just run cdk deploy. That's all!!!