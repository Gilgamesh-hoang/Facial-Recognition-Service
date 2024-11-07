import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
import logging
import src.utils.constant as constant
from src.services.preprocessing_service import remove_outliers

logger = logging.getLogger(__name__)

def load_model_from_file() -> SGDClassifier | None:
    try:
        if os.path.exists(constant.CLASSIFY_MODEL_PATH):
            with open(constant.CLASSIFY_MODEL_PATH, 'rb') as file:
                model = pickle.load(file)
            logger.info('Model loaded successfully')
        else:
            model = create_and_save_face_model()  # Gọi hàm tạo model mới nếu file không tồn tại

        return model  # Trả về model sau khi đã gán giá trị
    except FileNotFoundError:
        logger.error('Model file not found')
        return None  # Trả về None nếu có lỗi


def save_model_to_file(model: SGDClassifier):
    if model is None:
        logger.error('Model is None')
        return

    with open(constant.CLASSIFY_MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    logger.info('Model saved successfully')


def create_and_save_face_model() -> SGDClassifier:
    with open(constant.DATASET_EMBEDDINGS_PATH, 'rb') as f:
        (emb_array, labels) = pickle.load(f)

    X = np.array(emb_array, dtype=np.float64)
    y = np.array(labels)

    # Mã hóa nhãn thành dạng số
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    logger.info('Train model with dummy data')
    # Khởi tạo và huấn luyện lần đầu
    model_classify = SGDClassifier(alpha=0.0001, learning_rate='optimal', loss='log_loss', max_iter=1000, penalty='l2',
                                   random_state=42)
    # Chỉ định các lớp cần phân loại
    classes = np.arange(1000)  # 1000 lớp
    model_classify.partial_fit(X, y_encoded, classes=classes)

    save_label_encode_file(label_encoder)

    logger.info('Model initialized')
    return model_classify


async def train_model(service, user_id: str, embeddings: list[np.ndarray]):
    """
    Continue training the loaded model with new data.

    Parameters:
    - userId: The user ID as a label for the embeddings.
    - imagesData: List of images (in bytes) to generate embeddings for training.
    """
    cleaned_embeddings = remove_outliers(embeddings)
    if cleaned_embeddings is None:
        logger.error('No embeddings to train')
        return

    X_new = [embedding.flatten() for embedding in cleaned_embeddings]  # Flatten embeddings
    X_new = np.array(X_new, dtype=np.float64)
    y_new = np.array([user_id] * len(X_new))  # Mảng chứa các nhãn giống nhau

    label_encoder = service.get_label_encoder()
    # Lấy các nhãn hiện tại và thêm nhãn mới
    all_labels = np.unique(list(label_encoder.classes_) + [user_id])
    # Mã hóa nhãn thành dạng số
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)

    # Mã hóa nhãn mới
    y_new_encoded = label_encoder.transform(y_new)
    model = service.get_model()
    # Huấn luyện cải thiện với dữ liệu mới
    model.partial_fit(X_new, y_new_encoded)
    logger.info('Model trained successfully with user ID: %s', user_id)
    service.set_label_encoder(label_encoder)


def predict_model(service, embeddings: list[np.ndarray]) -> list[str]:
    model = service.get_model()
    label_encoder = service.get_label_encoder()
    X = [embedding.flatten() for embedding in embeddings]
    X = np.array(X, dtype=np.float64)
    y_pred = model.predict(X)
    user_ids = label_encoder.inverse_transform(y_pred)
    return user_ids.tolist()


def load_label_encode_from_file() -> LabelEncoder:
    if os.path.exists(constant.LABEL_ENCODE_PATH):
        with open(constant.LABEL_ENCODE_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
            logger.info('Label encoder loaded successfully')
            return label_encoder
    else:
        raise ValueError('Label encoder file not found')


def save_label_encode_file(label_encoder: LabelEncoder):
    with open(constant.LABEL_ENCODE_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)
    logger.info('Label encoder saved successfully')


if __name__ == '__main__':
    pass
