import os
import yaml


def yaml_config_hook(config_file):
    """
    Custom YAML config loader, which can include other yaml files
    """

    # load yaml files in the nested 'defaults' section, which include defaults for experiments
    with open(config_file) as f:
        cfg = yaml.safe_load(f)
        for d in cfg.get("defaults", []):
            config_dir, cf = d.popitem()
            cf = os.path.join(os.path.dirname(config_file), config_dir, cf + ".yaml")
            with open(cf) as f:
                l = yaml.safe_load(f)
                cfg.update(l)

    if "defaults" in cfg.keys():
        del cfg["defaults"]

    return cfg


class Tee:
    def __init__(self, fname, stream, mode="a+"):
        self.stream = stream
        self.file = open(fname, mode)

    def write(self, message):
        self.stream.write(message)
        self.file.write(message)
        self.flush()

    def print(self, message):
        self.write(message + "\n")

    def flush(self):
        self.stream.flush()
        self.file.flush()


# replaces marker "STORAGE_PATH" with environment variable "STORAGE_PATH" or "." if not defined
def add_mount_path_if_required(path):
    return path.replace("STORAGE_PATH", os.environ.get('STORAGE_PATH', "."))
    
