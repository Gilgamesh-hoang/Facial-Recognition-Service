import os
import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder

from src.services.model_service import ModelService

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'face-model.pkl')
LABEL_ENCODE_PATH = os.path.join(BASE_DIR, 'Models', 'label-encode.pkl')


def load_model_from_file() -> SGDClassifier | None:
    try:
        if os.path.exists(FACE_MODEL_PATH):
            with open(FACE_MODEL_PATH, 'rb') as file:
                model = pickle.load(file)
            print('Model loaded successfully')
        else:
            model = create_and_save_face_model()  # Gọi hàm tạo model mới nếu file không tồn tại

        return model  # Trả về model sau khi đã gán giá trị
    except FileNotFoundError:
        print('Model file not found')
        return None  # Trả về None nếu có lỗi


def save_model_to_file(model: SGDClassifier):
    if model is None:
        print('Model is None')
        return

    with open(FACE_MODEL_PATH, 'wb') as file:
        pickle.dump(model, file)
    print('Model saved successfully')


def create_and_save_face_model() -> SGDClassifier:
    with open(os.path.join(BASE_DIR, 'Dataset', 'FaceData', 'embeddings.pkl'), 'rb') as f:
        data = pickle.load(f)

    # Chuẩn bị dữ liệu
    X, y = [], []  # Embeddings và nhãn

    for label, embeddings_list in data.items():
        for embedding in embeddings_list:
            X.append(embedding.flatten())
            y.append(label)

    X = np.array(X, dtype=np.float64)
    y = np.array(y)

    # Mã hóa nhãn thành dạng số
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print('Train model with dummy data')

    # Khởi tạo và huấn luyện lần đầu
    model_classify = SGDClassifier(alpha=0.0001, learning_rate='optimal', loss='log_loss', max_iter=1000, penalty='l2',
                                   random_state=42)
    # Chỉ định các lớp cần phân loại
    classes = np.arange(800)  # 800 lớp từ 0 đến 799
    model_classify.partial_fit(X, y_encoded, classes=classes)

    save_label_encode_file(label_encoder)

    print("Model initialized")
    return model_classify


def train_model(service: ModelService, user_id: str, embeddings: list[np.ndarray]):
    """
    Continue training the loaded model with new data.

    Parameters:
    - userId: The user ID as a label for the embeddings.
    - imagesData: List of images (in bytes) to generate embeddings for training.
    """
    X_new = [embedding.flatten() for embedding in embeddings]  # Flatten embeddings
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

    service.set_label_encoder(label_encoder)

def load_label_encode_from_file() -> LabelEncoder:
    if os.path.exists(LABEL_ENCODE_PATH):
        with open(LABEL_ENCODE_PATH, 'rb') as file:
            label_encoder = pickle.load(file)
            print('Label encoder loaded successfully')
            return label_encoder
    else:
        raise ValueError('Label encoder file not found')


def save_label_encode_file(label_encoder: LabelEncoder):
    with open(LABEL_ENCODE_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)
    print('Label encoder saved successfully')


if __name__ == '__main__':
    pass
