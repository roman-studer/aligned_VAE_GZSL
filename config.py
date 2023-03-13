from dataclasses import dataclass
import yaml


@dataclass
class Config:
    config_dir: str

    @classmethod
    def load_set_yaml(cls):
        return cls

    @classmethod
    def load_set_dict(cls):
        return cls

    def load_yaml(self, name):
        return yaml.safe_load(self.config_dir + name + '.yaml')

    @staticmethod
    def load_dict(name):
        pass


class Paths(Config):
    working_dir: str
    model_dir: str
    data_dir: str

    def __post_init__(self):
        self.validate_path_len()


    def from_yaml(self):
        return Config.load_set_yaml(self)

    def from_dict(self):
        return Config.load_set_dict(self)


    def validate_path_len(self):
        assert len(self.working_dir) < 260


class Image(Config):
    name: str = 'image'
    rgb_channels: tuple = (81, 131, 181)  # fake RGB channels, corresponding to red, green and blue channel

    def from_yaml(self):
        return Config.load_set_yaml(self)

    def from_dict(self):
        return Config.load_set_dict(self)

