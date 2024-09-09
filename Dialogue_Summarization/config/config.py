import yaml
from pprint import pprint

def load_config(config_path="/root/dialogue/Dialogue_prj/config/config.yaml"):
    """
<<<<<<< HEAD
    yaml 파일 로드
    
    Parameters:
    - config_path (str): config yaml
    
    Returns:
    - dict: 딕셔너리로 config 전달
=======
    Load the configuration from a YAML file.
    
    Parameters:
    - config_path (str): The path to the config.yaml file.
    
    Returns:
    - dict: Loaded configuration data as a Python dictionary.
>>>>>>> 54570e73c6867250070e456a3d275d63930bc1ff
    """
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    
    pprint(loaded_config)  # Print the loaded configuration for verification
    return loaded_config

# Example usage
if __name__ == "__main__":
    config = load_config()
