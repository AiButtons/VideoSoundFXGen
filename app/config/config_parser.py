import json, os

def get_config():
    config_data = None
    script_dir = os.path.dirname(__file__)
    with open(f'{script_dir}/config.json', 'r') as config_file:
        config_data = json.load(config_file)
        
    return config_data