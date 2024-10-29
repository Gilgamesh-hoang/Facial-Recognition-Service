import os
import pickle

import cv2
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

import src.face_recognition.facenet as facenet
from src.align import detect_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'face-model.pkl')
FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model


def load_model_from_file() -> SGDClassifier | None:
    model = None  # Khai báo model ở phạm vi rộng hơn
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

    print("Model initialized")
    return model_classify


def get_embeddings(imagesData: list[bytes]) -> list[np.ndarray]:
    # Configuration parameters for face detection and recognition
    MINSIZE = 20  # Minimum size of the face
    THRESHOLD = [0.7, 0.7, 0.8]  # Three steps' threshold
    FACTOR = 0.709  # Scale factor
    INPUT_IMAGE_SIZE = 160  # Size of the input image for the model

    # Create a new TensorFlow graph
    with tf.Graph().as_default():
        # Configure GPU options
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        # Create a new TensorFlow session with the specified GPU options
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        # Set the session as default
        with sess.as_default():
            # Load the FaceNet model
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors from the model
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Initialize MTCNN networks for face detection
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            embeddings_list = []
            for imageData in imagesData:
                # Decode the image bytes into an OpenCV frame
                frame = cv2.imdecode(np.frombuffer(imageData, np.uint8), cv2.IMREAD_COLOR)
                # Detect faces in the image
                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces_found = bounding_boxes.shape[0]  # Number of faces found
                if faces_found == 0 or faces_found > 1:
                    continue

                bounding_box_coordinates = bounding_boxes[:, 0:4]
                bounding_boxes_array = np.zeros((faces_found, 4), dtype=np.int32)

                # Extract bounding box coordinates for each face
                bounding_boxes_array[0][0] = bounding_box_coordinates[0][0]
                bounding_boxes_array[0][1] = bounding_box_coordinates[0][1]
                bounding_boxes_array[0][2] = bounding_box_coordinates[0][2]
                bounding_boxes_array[0][3] = bounding_box_coordinates[0][3]

                # Crop and resize each face
                cropped = frame[bounding_boxes_array[0][1]:bounding_boxes_array[0][3],
                          bounding_boxes_array[0][0]:bounding_boxes_array[0][2], :]
                resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)  # Prewhiten the image
                reshaped = prewhitened.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)  # Reshape for the model

                # Get the embedding vector for the face
                feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                embedding = sess.run(embeddings, feed_dict=feed_dict)  # Run the session to get the embedding

                embeddings_list.append(embedding)

            sess.close()  # Close the session

            return embeddings_list




def train_classifier(userId: str, images: list[np.ndarray]):
    """
    Continue training the loaded model with new data.

    Parameters:
    - userId: The user ID as a label for the embeddings.
    - imagesData: List of images (in bytes) to generate embeddings for training.

    Returns:
    - Status message indicating success or error.
    """
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\embedding1.pkl', 'rb') as f1:
        embedding1 = pickle.load(f1)
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\embedding2.pkl', 'rb') as f2:
        embedding2 = pickle.load(f2)

    embeddings = [embedding1, embedding2]
    model = GaussianNB(var_smoothing=1e-12)
    # Initialize with two dummy classes
    initial_classes = ['user1', 'user2']
    # initial_classes = np.array(['user1', 'user2'])
    model.fit(X=np.zeros((2, 512)), y=initial_classes)

    # Reshape embeddings to 2D array
    X_new = np.array(embeddings).reshape(len(embeddings), -1)
    y_new = np.array([userId] * len(embeddings))  # Use userId as label for all embeddings

    all_user_ids = ['user3', 'user3']
    # all_user_ids.append(userId)

    # Perform incremental training using partial_fit
    # model.partial_fit(X_new, y_new, classes=np.unique(np.array(all_user_ids)))
    model.partial_fit(X=np.zeros((2, 512)), y=all_user_ids, classes=np.unique(['user1', 'user2', 'user3']))

    return {"status": "success", "message": "Model updated and saved successfully"}

    # with open(file_path, 'rb') as f:
    #     data = pickle.load(f)




# def p2():
#     model, X_test, y_test, label_encoder1 = p1()
#
#     with open('E:\Facial-Recognition-Service\Dataset\FaceData\hoang_embeddings.pkl', 'rb') as f:
#         data = pickle.load(f)
#
#         # Chuẩn bị dữ liệu
#     X_new = []  # Embeddings và nhãn
#
#     for label, embeddings_list in data.items():
#         for embedding in embeddings_list:
#             X_new.append(embedding.flatten())
#
#     X_new = np.array(X_new, dtype=np.float64)
#     y_new = np.array(['hoang'] * len(X_new))
#
#     # Lấy các nhãn hiện tại và thêm nhãn mới
#     new_labels = np.unique(y_new)
#     all_labels = np.unique(list(label_encoder1.classes_) + list(new_labels))
#
#     # Mã hóa nhãn thành dạng số
#     label_encoder1 = LabelEncoder()
#     label_encoder1.fit(all_labels)
#
#     # Mã hóa nhãn mới
#     y_new_train_encoded = label_encoder1.transform(new_labels)
#     a = label_encoder1.transform(y_new)
#     print('y_new_train_encoded: ', y_new_train_encoded)
#     print(f"Learned labels: {len(list(label_encoder1.classes_))}")
#
#     # Chia tập dữ liệu thành train và test
#     X_train_new, X_test_new, y_train_new, y_test_new = train_test_split(X_new, a, test_size=0.2, random_state=42)
#     # Huấn luyện cải thiện với dữ liệu mới
#     # model.partial_fit(X_train_new, y_train_new)
#     model.partial_fit(X_new, a)
#


if __name__ == '__main__':
    # create_and_save_face_model()
    # Giả sử bạn có dữ liệu ảnh dưới dạng bytes
    # with (open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\21130363.png', 'rb') as f1,
    #       open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\DSC_0024.png', 'rb') as f2):
    #     images_data = [f1.read(), f2.read()]
    pass
