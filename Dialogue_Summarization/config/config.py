import yaml
from pprint import pprint

def load_config(config_path="/root/dialogue/Dialogue_prj/config/config.yaml"):
    """
    yaml 파일 로드
    
    Parameters:
    - config_path (str): config yaml
    
    Returns:
    - dict: 딕셔너리로 config 전달
    """
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    
    pprint(loaded_config)  # Print the loaded configuration for verification
    return loaded_config

# Example usage
if __name__ == "__main__":
    config = load_config()
