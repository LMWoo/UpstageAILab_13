from models.base import BaseModel

class TutorialModel(BaseModel):
    def __init__(self, data_preprocessor):
        super().__init__(data_preprocessor)
        print("Initialize Tutorial Model ")
    def encoding(self):
        pass

    def splitdata(self):
        pass

    def train(self):
        pass

    def validation(self):
        pass

    def test(self):
        pass

    def analysis_validation(self, save_path):
        pass

    def save_model(self, save_path):
        pass

    def load_model(self, load_path):
        pass