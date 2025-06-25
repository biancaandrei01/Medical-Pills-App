import yaml

def load_config(config_path="../resources/application.yaml"):
    """
    Loads the YAML configuration file.
    """
    try:
        with open(config_path, 'r') as file:
            # Use safe_load to avoid potential security vulnerabilities
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        print(f"Error: The configuration file '{config_path}' was not found.")
        return None
    except yaml.YAMLError as e:
        print(f"Error parsing the YAML file: {e}")
        return None