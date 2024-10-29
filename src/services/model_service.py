from sklearn.linear_model import SGDClassifier
import numpy as np
from src.services.classification import load_model_from_file, save_model_to_file


class ModelService:
    def __init__(self):
        self.model: SGDClassifier | None = None

    def predict_user(self, data) -> str:
        pass

    def load_model_from_file(self):
        self.model = load_model_from_file()

    def save_model_to_file(self):
        save_model_to_file(self.model)

    def train_classifier(self, user_id: str, images: list[np.ndarray]):
        print('user_id:', user_id)
        print('images size:', len(images))
        print('model is None:', self.model is None)

