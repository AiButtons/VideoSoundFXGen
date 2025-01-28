import gradio as gr
import boto3
import os
import requests
import tempfile
import uuid
from pathlib import Path
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.audio.io.AudioFileClip import AudioFileClip
import jwt

# Initialize configurations
s3_client = boto3.client('s3')
BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
API_KEY = os.getenv('API_KEY')
FASTVIDEO_ENDPOINT = os.getenv('FASTVIDEO_ENDPOINT')
MMAUDIO_ENDPOINT = os.getenv('MMAUDIO_ENDPOINT')

def generate_jwt_token(api_key):
    """Generate a JWT token that expires in 2 hours"""
    import jwt
    from datetime import datetime, timedelta
    
    # Create token payload
    payload = {
        'exp': datetime.utcnow() + timedelta(hours=2),  # Expires in 2 hours
        'iat': datetime.utcnow(),  # Issued at time
        'api_key': api_key
    }
    
    # Create token
    token = jwt.encode(payload, API_KEY, algorithm='HS256')
    return token

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
        jwt_token = generate_jwt_token(API_KEY)
        
        response = requests.post(
            FASTVIDEO_ENDPOINT,
            params=params,
            headers={"Authorization": f"Bearer {jwt_token}"}
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
        
        
        jwt_token = generate_jwt_token(API_KEY)
        
        response = requests.post(
            MMAUDIO_ENDPOINT,
            params=params,
            headers={"Authorization": f"Bearer {jwt_token}"}
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

# Create Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as app:
    gr.Markdown("# Video and Sound Effects Generator")
    
    # State variables
    video_url_state = gr.State()
    video_path_state = gr.State()
    
    # Video Generation Section
    with gr.Group():
        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown(
                    '<div style="background-color: #1f2937; padding: 10px; border-radius: 8px; margin-bottom: 20px;"><p style="color: aqua; margin: 0;"> Generate a video or upload a video, then generate the sound effects for that video</p></div>'
                )
                prompt_input = gr.Textbox(
                    label="Enter prompt for video generation",
                    placeholder="A mystical forest with floating lights..."
                )
                with gr.Row():
                    height_input = gr.Number(value=720, label="Height")
                    width_input = gr.Number(value=1280, label="Width")
                    num_frames_input = gr.Number(value=21, label="Number of Frames")
                with gr.Row():
                    seed_input = gr.Number(value=42, label="Seed")
                    fps_input = gr.Number(value=18, label="FPS")
                    num_inference_steps_input = gr.Number(value=10, label="Inference Steps")
                    flow_shift_input = gr.Number(value=17, label="Flow Shift")
                    
            with gr.Column(scale=2):
                video_upload = gr.Video(label="Upload Video")
                video_url_input = gr.Textbox(
                    label="Or Paste Video URL",
                    placeholder="https://example.com/video.mp4"
                )

                def handle_video_upload(video_file, video_url):
                    try:
                        if video_file:
                            url = upload_to_s3(video_file, 'video/mp4')
                            return {
                                video_output: video_file,
                                video_path_state: video_file,
                                video_url_state: url,
                                video_status: "Video uploaded successfully!",
                                sound_effects_group: gr.update(visible=True)
                            }
                        elif video_url:
                            temp_path = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
                            response = requests.get(video_url)
                            response.raise_for_status()
                            with open(temp_path, 'wb') as f:
                                f.write(response.content)
                            url = upload_to_s3(temp_path, 'video/mp4')
                            return {
                                video_output: temp_path,
                                video_path_state: temp_path,
                                video_url_state: url,
                                video_status: "Video processed successfully!",
                                sound_effects_group: gr.update(visible=True)
                            }
                        return {
                            video_output: None,
                            video_path_state: None,
                            video_url_state: None,
                            video_status: "Please upload a video or provide a URL",
                            sound_effects_group: gr.update(visible=False)
                        }
                    except Exception as e:
                        return {
                            video_output: None,
                            video_path_state: None,
                            video_url_state: None,
                            video_status: f"Error: {str(e)}",
                            sound_effects_group: gr.update(visible=False)
                        }

                # Add event handler for upload button
        generate_upload_btn = gr.Button("Generate / Upload Video", variant="primary")
        
        def handle_video_input(prompt, height, width, num_frames, seed, fps, steps, flow_shift, video_file, video_url):
            if video_file or video_url:
                return handle_video_upload(video_file, video_url)
            else:
                return handle_video_generation(prompt, height, width, num_frames, seed, fps, steps, flow_shift)
            
        video_output = gr.Video(label="Generated Video")
        video_status = gr.Markdown()

    # Sound Effects Section (Initially Hidden)
    with gr.Group(visible=False) as sound_effects_group:
        gr.Markdown("### Add Sound Effects")
        sound_prompt = gr.Textbox(
            label="Sound Effect Description (optional)",
            placeholder="Mystical ambient sounds..."
        )
        with gr.Row():
            cfg_strength = gr.Slider(
                minimum=1.0,
                maximum=10.0,
                value=4.5,
                label="CFG Strength"
            )
            sound_seed = gr.Number(value=42, label="Sound Seed")
        
        generate_sound_btn = gr.Button("Add Sound Effects", variant="primary")
        final_video = gr.Video(label="Final Video with Sound")
        sound_status = gr.Markdown()

    def handle_video_generation(prompt, height, width, num_frames, seed, fps, steps, flow_shift):
        try:
            video_path, status = process_video_generation(
                prompt, height, width, num_frames, seed, fps, steps, flow_shift
            )
            if video_path:
                url = upload_to_s3(video_path, 'video/mp4')  # We need this URL for MM Audio
                return {
                    video_output: video_path,
                    video_path_state: video_path,
                    video_url_state: url,  # This was missing in the outputs!
                    video_status: status,
                    sound_effects_group: gr.update(visible=True)
                }
            return {
                video_output: None,
                video_path_state: None,
                video_url_state: None,
                video_status: status,
                sound_effects_group: gr.update(visible=False)
            }
        except Exception as e:
            return {
                video_output: None,
                video_path_state: None,
                video_url_state: None,
                video_status: f"Error: {str(e)}",
                sound_effects_group: gr.update(visible=False)
            }

    def handle_sound_generation(video_url, prompt, cfg, seed):
        try:
            if not video_url:
                return None, "Please generate a video first"
                    
            # Generate sound effects and upload to S3
            audio_path = generate_sound_effects(
                video_url,
                prompt=prompt or "",
                cfg_strength=cfg,
                seed=seed
            )
            
            # Upload audio to S3 and get URL
            audio_url = upload_to_s3(audio_path, 'audio/flac')
            
            # Merge using both presigned URLs and get final video URL
            final_video_url = merge_video_audio(video_url, audio_url)
            
            # Clean up audio file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {audio_path}: {e}")
            
            return final_video_url, "Sound effects added successfully!"
        except Exception as e:
            return None, f"Error adding sound effects: {str(e)}"

    generate_upload_btn.click(
            handle_video_input,
            inputs=[
                prompt_input, height_input, width_input, num_frames_input,
                seed_input, fps_input, num_inference_steps_input, flow_shift_input,
                video_upload, video_url_input
            ],
            outputs=[video_output, video_path_state, video_url_state, video_status, sound_effects_group]
        )
    generate_sound_btn.click(
        handle_sound_generation,
        inputs=[
            video_url_state, 
            sound_prompt, cfg_strength, sound_seed
        ],
        outputs=[final_video, sound_status]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=8080, share=True)