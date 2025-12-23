import utility as util

class Config:
    def __init__(self):
        self.config = util.load_config("config.yml")

    def get_config(self):
        return self.config
    
    def get_model_name(self):
        return self.config['model_use']['name']

    def get_model_params(self):
        params = util.load_config('config/' + self.get_model_name() + '.yml')
        return params