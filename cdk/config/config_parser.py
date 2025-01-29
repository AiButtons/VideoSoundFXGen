import json, os

def get_config():
    config_data = {}
    script_dir = os.path.dirname(__file__)
    
    # Load config.json
    config_path = os.path.join(script_dir, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as config_file:
            config_data.update(json.load(config_file))
    
    # Load endpoints.json from modal directory
    endpoints_path = os.path.join(os.path.dirname(os.path.dirname(script_dir)), 'modal', 'endpoints.json')
    if os.path.exists(endpoints_path):
        with open(endpoints_path, 'r') as endpoints_file:
            endpoints = json.load(endpoints_file)
            config_data['endpoints'] = {
                'FASTVIDEO_ENDPOINT': endpoints.get('fasthunyuan'),
                'MMAUDIO_ENDPOINT': endpoints.get('mmaudio')
            }
    
    return config_data