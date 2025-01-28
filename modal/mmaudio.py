import modal
from fastapi import Response, Depends, HTTPException, status, Security, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import os
import requests
import io
import traceback
import logging
import jwt
import tempfile
from pathlib import Path
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

cuda_version = "12.2.0"
flavor = "devel"
os_version = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{os_version}"

MODEL_VARIANT = "large_44k_v2"
MINUTES = 60
HOURS = 60 * MINUTES
MMA_GITHUB_URL = "https://github.com/hkchengrex/MMAudio.git" 
MODEL_VOLUME_NAME = "mmaudio_weights"
CACHE_VOLUME_NAME = "mmaudio_cache"
MODELS_PATH = Path('/root/MMAudio') 
CHECKPOINT_CACHE_PATH = Path("/root/.cache/torch/hub/checkpoints")

auth_scheme = HTTPBearer()

model_volume = modal.Volume.from_name(MODEL_VOLUME_NAME, create_if_missing=True)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)

def download_face_models():
    """Download face detection and alignment models"""
    import urllib.request
    import zipfile
    
    FACE_MODELS = {
        "s3fd": {
            "url": "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth",
            "filename": "s3fd-619a316812.pth"
        },
        "2dfan4": {
            "url": "https://www.adrianbulat.com/downloads/python-fan/2DFAN4-cd938726ad.zip",
            "filename": "2DFAN4-cd938726ad.zip"
        }
    }
    
    for model_name, model_info in FACE_MODELS.items():
        target_path = CHECKPOINT_CACHE_PATH / model_info["filename"]
        if not target_path.exists():
            logger.info(f"Downloading {model_name} model...")
            urllib.request.urlretrieve(model_info["url"], target_path)
            
            if model_info["filename"].endswith(".zip"):
                with zipfile.ZipFile(target_path, 'r') as zip_ref:
                    zip_ref.extractall(CHECKPOINT_CACHE_PATH)

def install_mmaudio():
    import subprocess
    import os
    import glob

    os.makedirs(CHECKPOINT_CACHE_PATH, exist_ok=True) 
    download_face_models()
    

    subprocess.run(["git", "clone", "--recurse-submodules", MMA_GITHUB_URL], 
                    check=True)
    os.chdir("MMAudio")
    py_files = glob.glob("mmaudio/**/*.py", recursive=True)
    
    # Patch each file
    for py_file in py_files:
        print(f"Patching {py_file}")
        with open(py_file, 'r') as f:
            content = f.read()
        content = content.replace("from mmaudio.", "from MMAudio.mmaudio.")
        content = content.replace("import mmaudio.", "import MMAudio.mmaudio.")
        with open(py_file, 'w') as f:
            f.write(content)
            
    subprocess.run(["pip", "install", "-v", "-e", "."], check=True)  

    from MMAudio.mmaudio.eval_utils import all_model_cfg
    model = all_model_cfg[MODEL_VARIANT]
    model.download_if_needed()

mmaudio_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.10"
    ).apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "ninja-build",
    ).pip_install(
        "packaging", 
        "ninja", 
        "wheel",  
        "setuptools",
        'numpy==1.26.2',
        "torchaudio>=2.5.1",
        "fastapi",
        "hf-transfer",
        "PyJWT",
        "requests",
    ).env({"HF_HUB_ENABLE_HF_TRANSFER": "1"}
    ).run_function(
        install_mmaudio,
        volumes={
            MODELS_PATH: model_volume,
            CHECKPOINT_CACHE_PATH: cache_volume
        }
        )
)

app = modal.App(name="mmaudio-synthesis")

@app.cls(
    image=mmaudio_image,
    gpu=modal.gpu.T4(count=1),
    timeout=1 * HOURS,
    container_idle_timeout=10 * MINUTES,
    volumes={
        MODELS_PATH: model_volume,        # Mount directly where MMAudio looks
        CHECKPOINT_CACHE_PATH: cache_volume
        },
    secrets=[modal.Secret.from_name("api-key")]
)
class Model:
        
    
    @modal.enter()     
    def intialize(self):
        import torch
        from MMAudio.mmaudio.model.flow_matching import FlowMatching
        from MMAudio.mmaudio.model.networks import get_my_mmaudio
        from MMAudio.mmaudio.model.utils.features_utils import FeaturesUtils
        from MMAudio.mmaudio.eval_utils import all_model_cfg
        
        os.chdir("MMAudio")
        
        torch.backends.cuda.matmul.allow_tf32 = True  
        torch.backends.cudnn.allow_tf32 = True        # Added from demo
        self.model = all_model_cfg[MODEL_VARIANT]
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        dtype = torch.bfloat16
        try:    
            with torch.inference_mode():
                self.net = get_my_mmaudio(self.model.model_name).to(device, dtype).eval()
                self.net.load_weights(torch.load(self.model.model_path, 
                                                 map_location=device, weights_only=True))
                
                self.feature_utils = FeaturesUtils(
                    tod_vae_ckpt=self.model.vae_path,
                    synchformer_ckpt=self.model.synchformer_ckpt,
                    enable_conditions=True,
                    mode=self.model.mode,
                    bigvgan_vocoder_ckpt=self.model.bigvgan_16k_path,
                    need_vae_encoder=False
                ).to(device, dtype).eval()
                
                self.fm = FlowMatching(min_sigma=0, inference_mode='euler', num_steps=25)
                self.seq_cfg = self.model.seq_cfg
                logger.info("MMAudio model initialized successfully")
            
        except Exception as e:
            error_msg = f"Error during model initialization: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
        

    @torch.inference_mode()
    def process_video(
        self,
        video_url: str,
        prompt: str = "",
        negative_prompt: str = "",
        duration: float = None,
        cfg_strength: float = 4.5,
        seed: int = 42,
    ):
        from MMAudio.mmaudio.eval_utils import load_video, generate
        import torch
        import torchaudio
        
        try:
            # Download video
            response = requests.get(video_url)
            if response.status_code != 200:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Failed to download video from provided URL"
                )
            
            # Save to temporary file
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video.write(response.content)
            temp_video.close()
            
                    
            with torch.inference_mode():
                device = next(self.net.parameters()).device  # Get device from model
                
                video_info = load_video(temp_video.name, duration)
                clip_frames = video_info.clip_frames.detach().clone().unsqueeze(0).to(device)
                sync_frames = video_info.sync_frames.detach().clone().unsqueeze(0).to(device)
                
                # Update duration from video
                duration = video_info.duration_sec
                self.seq_cfg.duration = duration
                self.net.update_seq_lengths(
                    self.seq_cfg.latent_seq_len,
                    self.seq_cfg.clip_seq_len,
                    self.seq_cfg.sync_seq_len
                )
                
                # Generate audio with device-matched generator
                rng = torch.Generator(device=device).manual_seed(seed)
                audios = generate(
                    clip_frames.clone(),
                    sync_frames.clone(),
                    [prompt],
                    negative_text=[negative_prompt],
                    feature_utils=self.feature_utils,
                    net=self.net,
                    fm=self.fm,
                    rng=rng,
                    cfg_strength=cfg_strength
                )
            
                # Convert to audio file
                audio = audios.float().cpu()[0]
                
                # Save to temporary file
                temp_audio = tempfile.NamedTemporaryFile(suffix='.flac', delete=False)
                torchaudio.save(temp_audio.name, audio, self.seq_cfg.sampling_rate)
            
            try:
                with open(temp_audio.name, 'rb') as file:
                    content = file.read()
                    return Response(
                        content=content,
                        media_type="audio/flac",
                        headers={
                            "Content-Disposition": "attachment; filename=output.flac",
                        }
                    )
            finally:
                if os.path.exists(temp_audio.name):
                    os.unlink(temp_audio.name)
                if os.path.exists(temp_video.name):
                    os.unlink(temp_video.name)
                    
        except Exception as e:
            error_msg = f"Error during processing: {str(e)}"
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
        video_url: str,
        token: HTTPAuthorizationCredentials = Depends(auth_scheme),
        prompt: str = "",
        negative_prompt: str = "",
        duration: float = 8.0,
        cfg_strength: float = 4.5,
        seed: int = 42,
    ):
        try:
            jwt.decode(token.credentials, os.environ["api_key"], algorithms=["HS256"])
        except jwt.InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        return self.process_video(
            video_url,
            prompt,
            negative_prompt,
            duration,
            cfg_strength,
            seed
        )