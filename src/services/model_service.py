from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import src.services.classification as classification
from src.models.response import Response
from src.services.face_service import get_embeddings


class ModelService:
    def __init__(self):
        self.__model: SGDClassifier | None = None
        self.__label_encoder: LabelEncoder | None = None

    def predict_user(self, data) -> str:
        pass

    def load_components(self):
        self.__model = classification.load_model_from_file()
        self.__label_encoder = classification.load_label_encode_from_file()

    def save_components(self):
        classification.save_model_to_file(self.__model)
        classification.save_label_encode_file(self.__label_encoder)

    def train_classifier(self, user_id: str, images: list[bytes]) -> Response:
        embeddings = get_embeddings(images)

        if len(embeddings) == 0:
            return Response(Response.FACE_NOT_FOUND, "No faces found")

        classification.train_model(self, user_id, embeddings)

        if len(embeddings) != len(images):
            return Response(Response.FACE_NOT_DETECTED, "Some faces are not detected")

        return Response(Response.STATUS_CODE_SUCCESS, "Model trained successfully")

    def get_model(self) -> SGDClassifier:
        return self.__model

    def set_label_encoder(self, label_encoder: LabelEncoder):
        self.__label_encoder = label_encoder

    def get_label_encoder(self) -> LabelEncoder:
        return self.__label_encoder
