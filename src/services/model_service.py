import asyncio
import logging

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

import src.services.classification as classification
from src.models.response import Response
from src.services.face_service import get_embeddings
from src.services.preprocessing_service import PreprocessingService


class ModelService:
    def __init__(self):
        self.__model: SGDClassifier | None = None
        self.__label_encoder: LabelEncoder | None = None
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s - [in %(name)s:%(funcName)s():%(lineno)d] - %(message)s')

    def load_components(self):
        self.__model = classification.load_model_from_file()
        self.__label_encoder = classification.load_label_encode_from_file()

    def save_components(self):
        classification.save_model_to_file(self.__model)
        classification.save_label_encode_file(self.__label_encoder)

    def predict_user(self, image: bytes) -> Response:
        image = np.frombuffer(image, np.uint8)
        pre_process = PreprocessingService(face_number_per_img=4)
        data_processed = pre_process.pre_process_image([image])

        face_found = len(data_processed)
        if face_found == 0:
            return Response(Response.FACE_NOT_FOUND, "No faces found")
        elif face_found > 4:
            return Response(Response.MULTIPLE_FACES_FOUND, "Multiple faces found")

        embeddings = get_embeddings(data_processed)

        user_ids = classification.predict_model(self, embeddings)
        if user_ids is None:
            return Response(Response.USER_NOT_FOUND, "User not found")

        # user_ids = [uid for uid in user_ids if util.is_valid_uuid(uid)]
        return Response(
            Response.STATUS_CODE_SUCCESS,
            "User found",
            data=user_ids
        )

    def train_classifier(self, user_id: str, images: list[bytes]) -> Response:
        images = [np.frombuffer(img, np.uint8) for img in images]
        pre_process = PreprocessingService(face_number_per_img=1)
        data_processed = pre_process.pre_process_image(images)
        if data_processed is None:
            return Response(Response.FACE_NOT_FOUND, "No faces found")

        embeddings = get_embeddings(data_processed)

        # Khởi tạo một task bất đồng bộ để train model trong nền
        asyncio.create_task(classification.train_model(self, user_id, embeddings))

        if len(embeddings) != len(images):
            return Response(Response.FACE_NOT_DETECTED, "Some faces are not detected")
        return Response(Response.STATUS_CODE_SUCCESS, "Model trained successfully")

    def get_model(self) -> SGDClassifier:
        return self.__model

    def set_label_encoder(self, label_encoder: LabelEncoder):
        self.__label_encoder = label_encoder

    def get_label_encoder(self) -> LabelEncoder:
        return self.__label_encoder
