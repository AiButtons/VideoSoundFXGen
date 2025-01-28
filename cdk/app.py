from aws_cdk import (
    aws_lambda as lambda_,
    aws_ecr as ecr,
    Stack,
    CfnOutput,
    Duration,
    CfnParameter,
    aws_s3 as s3,
    aws_iam as iam,
    RemovalPolicy,
)
from constructs import Construct
import os

class FastAPIFunctionStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)

        # Reference existing SAM-managed S3 bucket
        sam_artifact_bucket = s3.Bucket(
            self, "AibuttonsVideoArtifactBucket",
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                allowed_origins=['*'],  # Consider specifying domains here for production
                allowed_headers=['*'],
                exposed_headers=['ETag', 'Content-Type', 'Accept-Ranges', 'Content-Range']
            )],
        )


        # Reference existing ECR repository
        ecr_repository = ecr.Repository.from_repository_name(
            self, "ECRRepo",
            repository_name="aibuttonsvideo2437f3f2/fastapifunctionead79d0drepo"
        )

        current_dir = os.path.dirname(os.path.realpath(__file__))
        app_dir = os.path.join(os.path.dirname(current_dir), 'app')

        # Define Lambda function
        fastapi_function = lambda_.DockerImageFunction(
            self, "AibuttonsVideoFunction",
            code=lambda_.DockerImageCode.from_image_asset(
                directory=app_dir    # Path to the directory containing your Dockerfile
            ),
            memory_size=3002,
            timeout=Duration.seconds(899),
            environment={
                "AWS_LWA_INVOKE_MODE": "RESPONSE_STREAM",
                "S3_BUCKET_NAME": sam_artifact_bucket.bucket_name,
                #"OPENAI_API_KEY": openai_api_key.value_as_string,
                "JWT_TOKEN": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ2aWRlb19nZW5lcmF0aW9uIn0.Zkewz_CL3-k7ilnV_6xgOzRaKXXoz1PnrIejacrYqvw",
                "FASTVIDEO_ENDPOINT": "https://zennah-jewel--fasthunyuan-fastvideo-generate.modal.run", # from updated config
                "MMAUDIO_ENDPOINT" : "https://schooluniguide--mmaudio-synthesis-model-generate.modal.run"
            },
            tracing=lambda_.Tracing.ACTIVE,
            
        )
        sam_artifact_bucket.grant_read_write(fastapi_function)
        # Define Function URL
        function_url = fastapi_function.add_function_url(
            auth_type=lambda_.FunctionUrlAuthType.NONE,
            invoke_mode=lambda_.InvokeMode.RESPONSE_STREAM
        )

        # Outputs
        CfnOutput(
            self, "FastAPIFunctionUrlOutput",
            description="Function URL for FastAPI function",
            value=function_url.url
        )

        CfnOutput(
            self, "FastAPIFunctionOutput",
            description="FastAPI Lambda Function ARN",
            value=fastapi_function.function_arn
        )

        CfnOutput(
            self, "AibuttonsVideoBucket",
            description="Name of the S3 bucket for SAM artifacts",
            value=sam_artifact_bucket.bucket_name
        )

from aws_cdk import App

app = App()
FastAPIFunctionStack(app, "AibuttonsVideoGen")
app.synth()