import os
import yaml


class CreateDashboard():
    def __init__(self, config_path:str):
        self.config_path = config_path
    
    def run(self):
        with open(self.config_path, "r") as stream:
            try:
                self.data = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)
        with open('./artifacts.yaml', 'w') as file:
            documents = yaml.dump(self.data, file)   
        return os.system("streamlit run app.py")
        