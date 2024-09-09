import yaml
from pprint import pprint

def load_config(config_path="/root/dialogue/Dialogue_prj/config/config.yaml"):
    """
    Load the configuration from a YAML file.
    
    Parameters:
    - config_path (str): The path to the config.yaml file.
    
    Returns:
    - dict: Loaded configuration data as a Python dictionary.
    """
    with open(config_path, "r") as file:
        loaded_config = yaml.safe_load(file)
    
    pprint(loaded_config)  # Print the loaded configuration for verification
    return loaded_config

# Example usage
if __name__ == "__main__":
    config = load_config()
