from aws_cdk import (
    aws_lambda as lambda_,
    aws_ecr as ecr,
    #aws_lambda_python_alpha as _alambda,
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
from aws_cdk.aws_ecr_assets import DockerImageAsset
from config.config_parser import get_config 
from cdk_ecr_deployment import ECRDeployment, DockerImageName

config = get_config()
class FastAPIFunctionStack(Stack):

    def __init__(self, scope: Construct, id: str, **kwargs) -> None:
        super().__init__(scope, id, **kwargs)
        api_key = CfnParameter(
            self, "ApiKey",
            type="String",
            description="API Key for the Lambda function",
            no_echo=True
        )
        
        # Reference existing SAM-managed S3 bucket
        sam_artifact_bucket = s3.Bucket(
            self, "AibuttonsVideoArtifactBucket",
            bucket_name=config.get('s3', {}).get('bucket_name', 'videosoundfxgen-artifacts'),
            removal_policy=RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            cors=[s3.CorsRule(
                allowed_methods=[s3.HttpMethods.GET, s3.HttpMethods.HEAD],
                allowed_origins=['*'],  # Consider specifying domains here for production
                allowed_headers=['*'],
                exposed_headers=['ETag', 'Content-Type', 'Accept-Ranges', 'Content-Range']
            )],
        )

        current_dir = os.path.dirname(os.path.realpath(__file__))
        app_dir = os.path.join(os.path.dirname(current_dir), 'app')
        # implementation of cold start improvement using snap start, commented out the sections
        
        
        # fastapi_layer = _alambda.PythonLayerVersion(
        #     self, "FastAPILayer",
        #     entry=app_dir,
        #     compatible_architectures=[lambda_.Architecture.X86_64],
        #     compatible_runtimes=[lambda_.Runtime.PYTHON_3_12],
        #     bundling=_alambda.BundlingOptions(
        #         asset_excludes=[
        #             ".venv", 
        #             "venv", 
        #             "__pycache__",
        #             "*.pyc",
        #             "test",
        #             "tests",
        #             "docs",
        #             ".git",
        #             ".pytest_cache",
        #             ".mypy_cache"
        #         ],
        #     )
        # )
        
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
                "API_KEY": api_key.value_as_string,
                "FASTVIDEO_ENDPOINT": config.get('endpoints', {}).get('FASTVIDEO_ENDPOINT'),
                "MMAUDIO_ENDPOINT": config.get('endpoints', {}).get('MMAUDIO_ENDPOINT')
            },
            tracing=lambda_.Tracing.ACTIVE,
                    
        )
        # fastapi_function = lambda_.Function(
        #     self, "AibuttonsVideoFunction",
        #     runtime=lambda_.Runtime.PYTHON_3_12,
        #     handler="main.handler",
        #     code=lambda_.Code.from_asset(app_dir),
        #     layers=[fastapi_layer],
        #     memory_size=3002,
        #     timeout=Duration.seconds(899),
        #     environment={
        #         "S3_BUCKET_NAME": sam_artifact_bucket.bucket_name,
        #         "API_KEY": api_key.value_as_string,
        #         "FASTVIDEO_ENDPOINT": config.get('endpoints', {}).get('FASTVIDEO_ENDPOINT'),
        #         "MMAUDIO_ENDPOINT": config.get('endpoints', {}).get('MMAUDIO_ENDPOINT')
        #     },
        #     tracing=lambda_.Tracing.ACTIVE,
        #     snap_start=lambda_.SnapStartConf.ON_PUBLISHED_VERSIONS
        # )
        # version = fastapi_function.current_version
        
        # the only reason for commenting was the the size of the lambda layer which exceeded 250 mb
        
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
FastAPIFunctionStack(app, config.get('deployment', {}).get('name', 'AibuttonsVideoSoundFxGen_v2'))
app.synth()