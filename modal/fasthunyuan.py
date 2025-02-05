import modal
from fastapi import Response, Depends, HTTPException, status, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import jwt
import torch
import subprocess
from pathlib import Path
import argparse

REVISION_ID = "7e948fca38562e218ae34485e005956592d36d9b"
FASTVIDEO_REPO = "FastVideo/FastHunyuan-diffusers"
MODEL_PATH = Path("/models") 
MODEL_VOLUME_NAME = "fasthunyuan-model"
MINUTES = 60
HOURS = 60 * MINUTES
NUM_GPUS = 1
DIFFUSERS_COMMIT = "https://github.com/huggingface/diffusers.git@91008aabc4b8dbd96a356ab6f457f3bd84b10e8b"

auth_scheme = HTTPBearer()

cuda_version = "12.2.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)


def download_model():
    from huggingface_hub import snapshot_download
    
    model_volume.reload()
    print("📥 Downloading FastHunyuan model...")
    
    model_path = MODEL_PATH / "FastHunyuan"
    if not model_path.exists():
        print("Cache miss - downloading model...")
        snapshot_download(
            repo_id=FASTVIDEO_REPO,
            revision=REVISION_ID,
            local_dir=str(model_path),
            repo_type="model",
        )
        model_volume.commit()
    else:
        print("Using cached model")

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10"
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
    ).pip_install("packaging", "ninja", "torch==2.5.0", "wheel",  "setuptools"
    ).pip_install("flash-attn==2.7.0.post2", extra_options="--no-build-isolation"
    ).pip_install(
    f"git+{DIFFUSERS_COMMIT}",
    "huggingface_hub",
    "transformers",
    "safetensors",
    "accelerate",
    "hf_transfer",
    "bitsandbytes==0.45.0",
    "moviepy==2.1.1",
    "SwissArmyTransformer>=0.4.12",
    "imageio",
    "loguru",
    "imageio-ffmpeg",
    "einops",
    "PyJWT",
    "pillow==9.5.0",
    "fastapi==0.115.6",
    "scikit-video",
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).run_function(
        download_model,
        volumes={MODEL_PATH: model_volume})
)


app = modal.App(name="fasthunyuan", image=image)

@app.cls(
    gpu=modal.gpu.A100(size='80GB', count=NUM_GPUS),
    timeout=1 * HOURS,
    container_idle_timeout=10 * MINUTES,
    volumes={MODEL_PATH: model_volume},
    secrets=[modal.Secret.from_name("api-key")],
    enable_memory_snapshot=True
)
class FastVideo:
    @modal.enter()    
    def initialize(self):
        
        from diffusers import (
            BitsAndBytesConfig, 
            HunyuanVideoPipeline,     
            HunyuanVideoTransformer3DModel)
        import torch
        from transformers.utils import move_cache
        
        model_volume.reload()
        
        model_path = MODEL_PATH / "FastHunyuan"
        if not model_path.exists():
            print("Model not found in volume - downloading...")
            self.download_model()
        
        move_cache()

        print("Using 8-bit quantization")
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_skip_modules=["proj_out", "norm_out"]
        )
        self.transformer = HunyuanVideoTransformer3DModel.from_pretrained(
            str(model_path),
            subfolder="transformer/",
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config #for quantization
        )  

        self.pipe = HunyuanVideoPipeline.from_pretrained(
            str(model_path),
            transformer=self.transformer,
            torch_dtype=torch.bfloat16
        )
        self.pipe.vae.enable_tiling()
        self.pipe.enable_model_cpu_offload()

        self.prompt_template = {
            "template": (
                "<|start_header_cid|>system<|end_header_id|>\n\nDescribe the video by detailing the following aspects: "
                "1. The main content and theme of the video."
                "2. The color, shape, size, texture, quantity, text, and spatial relationships of the contents, including objects, people, and anything else."
                "3. Actions, events, behaviors temporal relationships, physical movement changes of the contents."
                "4. Background environment, light, style, atmosphere, and qualities."
                "5. Camera angles, movements, and transitions used in the video."
                "6. Thematic and aesthetic concepts associated with the scene, i.e. realistic, futuristic, fairy tale, etc<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
            ),
            "crop_start": 95,
        }

    def _inference(
        self,
        prompt: str,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 45,
        seed: int = 1024,
        fps: int = 24,
        num_inference_steps: int = 6,
        flow_shift: int = 17,
    ):
        import io
        from diffusers.utils import export_to_video

        self.pipe.scheduler._shift = flow_shift
        generator = torch.Generator("cpu").manual_seed(seed)
        torch.cuda.reset_max_memory_allocated("cuda")
        video_frames = self.pipe(
            prompt=prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            prompt_template=self.prompt_template,
            num_inference_steps=num_inference_steps,
            generator=generator,
        ).frames[0]   
        
        output_path = "/tmp/output.mp4"
        export_to_video(video_frames, output_path, fps=fps)
        with open(output_path, "rb") as f:
            return io.BytesIO(f.read())

    @modal.web_endpoint(method="POST", docs=True)
    async def generate(
        self,
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
        prompt: str = "A camera panning through a beautiful landscape",
        height: int = 720,
        width: int = 1280,
        num_frames: int = 45,
        seed: int = 1024,
        fps: int = 24,
        num_inference_steps: int = 6,
        flow_shift: int = 17,
    ):
        try:
            jwt.decode(token.credentials, os.environ["api_key"], algorithms=["HS256"])
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return Response(
            content=self._inference(
                prompt,
                height,
                width,
                num_frames,
                seed,
                fps,
                num_inference_steps,
                flow_shift,
            ).getvalue(),
            media_type="video/mp4",
        )