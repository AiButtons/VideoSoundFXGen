import jwt
import time
import boto3
import os
import requests
import tempfile
import uuid
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip

def generate_jwt_token(api_key):
    """Generate JWT token using API key"""
    try:
        payload = {
            'exp': int(time.time()) + 7200,  # Token expires in 2 hour
            'iat': int(time.time()),
            'api_key': api_key
        }
        token = jwt.encode(payload, api_key, algorithm='HS256')
        return token
    except Exception as e:
        raise Exception(f"Failed to generate JWT token: {str(e)}")

s3_client = boto3.client('s3')
API_KEY = os.getenv('API_KEY')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
JWT_TOKEN = generate_jwt_token(API_KEY)
FASTVIDEO_ENDPOINT = os.getenv('FASTVIDEO_ENDPOINT')
MMAUDIO_ENDPOINT = os.getenv('MMAUDIO_ENDPOINT')


    
def adjust_frames_to_4k_plus_1(num_frames):
    remainder = (num_frames - 1) % 4
    adjusted = num_frames - remainder if remainder <= 2 else num_frames + (4 - remainder)
    if adjusted % 4 != 1:
        adjusted = adjusted - (adjusted % 4) + 1
    return adjusted

def merge_video_audio(video_url, audio_url):
    """Merge video and audio from URLs using moviepy"""
    temp_files = []
    try:
        # Create temp directory to ensure we have a valid working directory
        temp_dir = tempfile.mkdtemp()
        
        # Create moviepy temp directory
        moviepy_temp_dir = os.path.join(temp_dir, 'moviepy_temp')
        os.makedirs(moviepy_temp_dir, exist_ok=True)
        os.environ['MOVIEPY_TEMP_DIR'] = moviepy_temp_dir
        
        # Download video from URL
        video_response = requests.get(video_url)
        video_response.raise_for_status()
        temp_video = os.path.join(temp_dir, 'temp_video.mp4')
        with open(temp_video, 'wb') as f:
            f.write(video_response.content)
        temp_files.append(temp_video)

        # Download audio from URL
        audio_response = requests.get(audio_url)
        audio_response.raise_for_status()
        temp_audio = os.path.join(temp_dir, 'temp_audio.flac')
        with open(temp_audio, 'wb') as f:
            f.write(audio_response.content)
        temp_files.append(temp_audio)
        
        # Load clips
        video = VideoFileClip(temp_video)
        audio = AudioFileClip(temp_audio)
        
        # Combine video with audio
        final_clip = video.with_audio(audio)
        
        # Save result to temp directory
        output_path = os.path.join(temp_dir, 'output.mp4')
        final_clip.write_videofile(
            output_path,
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=os.path.join(moviepy_temp_dir, "temp-audio.m4a"),
            audio_bitrate="128k",
            threads=4,
            preset='ultrafast',
            ffmpeg_params=["-strict", "-2"]
        )
        temp_files.append(output_path)
        
        # Immediately check file and upload to S3
        if not os.path.exists(output_path):
            raise Exception("Video file was not created successfully")
            
        # Upload to S3
        file_key = f"{uuid.uuid4()}.mp4"
        s3_client.upload_file(
            output_path,
            BUCKET_NAME,
            file_key,
            ExtraArgs={'ContentType': 'video/mp4'}
        )
        
        # Generate URL
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': file_key},
            ExpiresIn=3600
        )
        
        # Close clips
        video.close()
        audio.close()
        final_clip.close()
            
        return url
        
    except Exception as e:
        raise Exception(f"Error merging video and audio: {str(e)}")
    finally:
        # Clean up all temporary files
        for file_path in temp_files:
            try:
                if os.path.exists(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {file_path}: {e}")
        # Remove temp directories
        try:
            if os.path.exists(moviepy_temp_dir):
                os.rmdir(moviepy_temp_dir)
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as e:
            print(f"Warning: Failed to delete temporary directory: {e}")
            

def process_video_generation(prompt, height, width, num_frames, seed, fps, num_inference_steps, flow_shift):
    try:
        adjusted_frames = adjust_frames_to_4k_plus_1(num_frames)
        params = {
            "prompt": prompt,
            "height": height,
            "width": width,
            "num_frames": adjusted_frames,
            "seed": seed,
            "fps": fps,
            "num_inference_steps": num_inference_steps,
            "flow_shift": flow_shift
        }
        
        response = requests.post(
            FASTVIDEO_ENDPOINT,
            params=params,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"}
        )
        response.raise_for_status()
        
        temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return temp_path, f"Video generated! Frames adjusted from {num_frames} to {adjusted_frames}"
    except Exception as e:
        return None, f"Error generating video: {str(e)}"

def generate_sound_effects(video_url, prompt="", cfg_strength=4.5, seed=42):
    try:
        params = {
            "video_url": video_url,
            "prompt": prompt,
            "negative_prompt": "voice",
            "cfg_strength": cfg_strength,
            "seed": seed
        }
        
        response = requests.post(
            MMAUDIO_ENDPOINT,
            params=params,
            headers={"Authorization": f"Bearer {JWT_TOKEN}"}
        )
        response.raise_for_status()
        
        temp_path = tempfile.NamedTemporaryFile(suffix='.flac', delete=False).name
        with open(temp_path, 'wb') as f:
            f.write(response.content)
        return temp_path
    except Exception as e:
        raise Exception(f"Error generating sound effects: {str(e)}")

def upload_to_s3(file_path, content_type):
    try:
        file_key = f"{uuid.uuid4()}{Path(file_path).suffix}"
        s3_client.upload_file(
            file_path,
            BUCKET_NAME,
            file_key,
            ExtraArgs={'ContentType': content_type}
        )
        url = s3_client.generate_presigned_url(
            'get_object',
            Params={'Bucket': BUCKET_NAME, 'Key': file_key},
            ExpiresIn=3600
        )
        return url
    except Exception as e:
        raise Exception(f"Error uploading to S3: {str(e)}")
