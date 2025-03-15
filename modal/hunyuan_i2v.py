import modal
from fastapi import Response, Depends, HTTPException, status, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import jwt
import torch
from pathlib import Path
import logging
import requests
import tempfile
import traceback
from PIL import Image
import io

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REVISION_ID = "3914f209367854b5e470f062c33159d5ab139e1e"
HUNYUAN_I2V_REPO = "tencent/HunyuanVideo-I2V"
MODEL_PATH = Path("/models") 
MODEL_VOLUME_NAME = "hunyuan-i2v"
MINUTES = 60
HOURS = 60 * MINUTES
NUM_GPUS = 1

GITHUB_REPO_URL  = "https://github.com/Tencent/HunyuanVideo-I2V.git"

auth_scheme = HTTPBearer()

cuda_version = "12.4.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


def download_model():
    from huggingface_hub import snapshot_download
    from transformers.utils import move_cache
    
    model_volume.reload()
    print("📥 Downloading Hunyuan I2V model...")
    
    model_path = MODEL_PATH / "Hunyuan-i2v"
    if not model_path.exists():
        print("Cache miss - downloading model...")
        snapshot_download(
            repo_id=HUNYUAN_I2V_REPO,
            revision=REVISION_ID,
            local_dir=str(model_path),
            repo_type="model",
        )
        model_volume.commit()
        move_cache()
    else:
        print("Using cached model")

def clone_repository():
    import subprocess
    subprocess.run(["git", "clone", "--recurse-submodules", GITHUB_REPO_URL, '/hunyuan-i2v'], check=True)

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11"
    ).apt_install(
        "libgl1-mesa-glx",
        "libglib2.0-0",
        "git",
        "cmake",
        "build-essential",
        "libgomp1",
        "libxrender1", 
        "libxext6",
        "libsm6",
        "ffmpeg",
    ).pip_install("packaging", "ninja", "wheel",  "setuptools", "torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0"
    ).pip_install("git+https://github.com/Dao-AILab/flash-attention.git@v2.6.3", extra_options="--no-build-isolation"
    ).pip_install(
        "huggingface_hub",
        "hf_transfer",
        #"moviepy==2.1.1",
        "PyJWT",
        "opencv-python==4.9.0.80",
        "diffusers==0.31.0",
        "accelerate==1.1.1",
        "pandas==2.0.3",
        "numpy==1.24.4",
        "einops==0.7.0",
        "tqdm==4.66.2",
        "loguru==0.7.2",
        "imageio==2.34.0",
        "imageio-ffmpeg==0.5.1",
        "safetensors==0.4.3",
        "peft==0.13.2",
        "transformers==4.39.3",
        "tokenizers==0.15.0",
        "deepspeed==0.15.1",
        "pyarrow==14.0.1",
        "tensorboard==2.19.0",
        "git+https://github.com/openai/CLIP.git",
        "fastapi==0.115.6",
        "xfuser==0.4.0",
        "yunchang>=0.4.0"
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).run_function(
        clone_repository,
        gpu=modal.gpu.A100(size='80GB'),
    ).run_function(
        download_model,
        volumes={MODEL_PATH: model_volume}
    )
)


app = modal.App(name="hunyuan-i2v", image=image)

@app.cls(
    gpu=modal.gpu.A100(size='80GB', count=NUM_GPUS),
    timeout=1 * HOURS,
    container_idle_timeout=1 * MINUTES,
    volumes={MODEL_PATH: model_volume},
    secrets=[modal.Secret.from_name("api-key")],
    enable_memory_snapshot=True
)
class HunyuanI2V:
    def __init__(self):
        self.sampler = None
        

    @modal.build()
    @modal.enter()
    def initialize(self):
        import os
        os.makedirs("/root/.triton/autotune", exist_ok=True)
        import sys
        sys.path.append('/hunyuan-i2v')
        
        try:
            logger.info("Loading Hunyuan I2V pipeline...")
            from hyvideo.inference import HunyuanVideoSampler
            from hyvideo.config import parse_args
            import sys
            
            # Set the correct model weight paths
            dit_weight_path = str(MODEL_PATH / "Hunyuan-i2v/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt")
            i2v_dit_weight_path = str(MODEL_PATH / "Hunyuan-i2v/hunyuan-video-i2v-720p/transformers/mp_rank_00_model_states.pt")
            
            # Create minimal args directly with command line arguments including correct paths
            sys.argv = [
                "script.py",
                "--model", "HYVideo-T/2",
                "--model-base", str(MODEL_PATH / "Hunyuan-i2v"),
                "--i2v-mode",
                "--i2v-resolution", "720p",
                "--i2v-condition-type", "token_replace",
                "--flow-shift", "7.0",
                "--flow-reverse",
                "--infer-steps", "50",
                "--i2v-stability",
                "--video-length", "45",
                "--video-size", "720", "1280",
                "--seed", "42",
                "--dit-weight", dit_weight_path,
                "--i2v-dit-weight", i2v_dit_weight_path
            ]
            
            # Let parse_args handle all the defaults and validations
            args = parse_args()
            
            # Initialize the model
            self.sampler = HunyuanVideoSampler.from_pretrained(MODEL_PATH / "Hunyuan-i2v", args=args)
            logger.info("Hunyuan I2V model initialized successfully")
        except Exception as e:
            error_msg = f"Error during model initialization: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

 
    def generate_video(
        self,
        prompt: str,
        image_url: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 45,
        seed: int = 42,
        fps: int = 24,
        num_inference_steps: int = 6,
        flow_shift: float = 7.0,
        i2v_stability: bool = True,
    ):
        try:
            # Download and process image
            response = requests.get(image_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to download image"
                )
            
            # Save image to a temporary file
            temp_img = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            temp_img_path = temp_img.name
            with open(temp_img_path, 'wb') as f:
                f.write(response.content)
            temp_img.close()
            
            # Generate video
            outputs = self.sampler.predict(
                prompt=prompt,
                height=height,
                width=width,
                video_length=num_frames,
                seed=seed,
                infer_steps=num_inference_steps,
                flow_shift=flow_shift,
                i2v_mode=True,
                i2v_image_path=temp_img_path,
                i2v_stability=i2v_stability
            )
            
            samples = outputs['samples']
            
            # Save the first generated video to a temporary file
            from hyvideo.utils.file_utils import save_videos_grid
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_path = temp_video.name
            temp_video.close()
            
            save_videos_grid(samples[0].unsqueeze(0), temp_path, fps=fps)
            
            try:
                with open(temp_path, 'rb') as file:
                    content = file.read()
                    return Response(
                        content=content,
                        media_type="video/mp4",
                        headers={
                            "Content-Disposition": "attachment; filename=output.mp4",
                        }
                    )
            finally:
                # Clean up temp files
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                if os.path.exists(temp_img_path):
                    os.unlink(temp_img_path)

        except Exception as e:
            error_msg = f"Error during video generation: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=error_msg
            )
    
    @modal.web_endpoint(method="POST", docs=True)
    async def generate(
        self,
        request: Request,
        prompt: str,
        image_url: str,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
        height: int = 720,
        width: int = 1280,
        num_frames: int = 45,
        seed: int = 42,
        fps: int = 24, 
        num_inference_steps: int = 6,
        flow_shift: float = 7.0,
        i2v_stability: bool = True
    ):
        # Verify JWT token
        try:
            jwt.decode(token.credentials, os.environ["zennah_api_key"], algorithms=["HS256"])
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return self.generate_video(
            prompt,
            image_url,
            height,
            width,
            num_frames,
            seed,
            fps,
            num_inference_steps,
            flow_shift,
            i2v_stability
        )