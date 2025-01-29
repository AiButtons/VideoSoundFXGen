import gradio as gr
import boto3
import os
import requests
import tempfile
from utils.utils import (
    process_video_generation,
    generate_sound_effects,
    merge_video_audio,
    upload_to_s3
)

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
            
            # Download the final video to a temporary file
            temp_final = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
            response = requests.get(final_video_url)
            response.raise_for_status()
            with open(temp_final, 'wb') as f:
                f.write(response.content)
            os.chmod(temp_final, 0o644)
            # Clean up audio file
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
            except Exception as e:
                print(f"Warning: Failed to delete temporary file {audio_path}: {e}")
            
            return temp_final, "Sound effects added successfully!"
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
    app.launch(server_name="0.0.0.0", server_port=8080, share=False)