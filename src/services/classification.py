import os
import pickle
import cv2
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier

import src.face_recognition.facenet as facenet
from src.align import detect_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'facemodel.pkl')
FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model


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


def load_face_model() -> SGDClassifier:
    if os.path.exists(FACE_MODEL_PATH):
        with open(FACE_MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
            return model
    else:
        return None


def create_and_save_face_model() -> SGDClassifier:
    model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42, tol=0.01)
    # model.coef_ = np.zeros((2, 512))
    # model.intercept_ = np.zeros(2)

    # Initialize with two dummy classes
    initial_classes = np.array(['user1', 'user2'])
    model.partial_fit(X=np.zeros((2, 512)), y=initial_classes, classes=np.unique(initial_classes))

    # with open(FACE_MODEL_PATH, 'wb') as file:
    #     pickle.dump(model, file)
    return model


# def train_classifier(userId: str, imagesData: list[bytes]):
#     """
#     Continue training the loaded model with new data.
#
#     Parameters:
#     - userId: The user ID as a label for the embeddings.
#     - imagesData: List of images (in bytes) to generate embeddings for training.
#
#     Returns:
#     - Status message indicating success or error.
#     """
#     # model = load_face_model()
#     # if not model:
#     #     return {
#     #         "status": "error",
#     #         "message": "Model not found"
#     #     }
#     #
#     embeddings = get_embeddings(imagesData)
#     # if not embeddings:
#     #     return {
#     #         "status": "error",
#     #         "message": "No faces found in the image"
#     #     }
#
#     model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=5, random_state=42, tol=0.01)
#
#     # Initialize with two dummy classes
#     initial_classes = ['user1', 'user2']
#     # initial_classes = np.array(['user1', 'user2'])
#     model.partial_fit(X=np.zeros((2, 512)), y=initial_classes, classes=np.unique(initial_classes))
#
#     # Reshape embeddings to 2D array
#     X_new = np.array(embeddings).reshape(len(embeddings), -1)
#     y_new = np.array([userId] * len(embeddings)) # Use userId as label for all embeddings
#
#
#     # Load existing classes from the model
#     # all_user_ids = model.classes_.tolist()
#
#     # if userId not in all_user_ids:
#     # all_user_ids.append(userId)
#     # print('All user ids:', all_user_ids)
#     # Perform incremental training using partial_fit
#     # model.partial_fit(X_new, y_new)
#     # model.partial_fit(X_new, y_new, classes=np.unique(all_user_ids))
#
#
#     return {"status": "success", "message": "Model updated and saved successfully"}
def train_classifier(userId: str):
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
    model = SGDClassifier(alpha=0.0001, learning_rate='optimal', loss='log_loss', max_iter=1000, penalty='l2',
                          random_state=42)
    # Initialize with two dummy classes
    initial_classes = ['user1', 'user2']
    # initial_classes = np.array(['user1', 'user2'])
    model.partial_fit(X=np.zeros((2, 512)), y=initial_classes, classes=np.unique(initial_classes))

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

def test1():
    user1_vector = np.random.rand(512)
    user2_vector = np.random.rand(512)

    X_train = np.array([user1_vector, user2_vector])
    y_train = np.array(['user1', 'user2'])

    model = SGDClassifier(loss='log_loss', penalty='l2', max_iter=1000, random_state=42, tol=0.01)
    model.partial_fit(X_train, y_train, classes=np.unique(y_train))

    user3_vector_1 = np.random.rand(512)
    user3_vector_2 = np.random.rand(512)

    X_new = np.array([user3_vector_1, user3_vector_2])
    y_new = np.array(['user2', 'user2'])

    model.partial_fit(X_new, y_new)

    # Kiểm tra dự đoán
    prediction = model.predict(X_new)
    print("Dự đoán cho user3:", prediction)

if __name__ == '__main__':
    # create_and_save_face_model()
    # Giả sử bạn có dữ liệu ảnh dưới dạng bytes
    # with (open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\21130363.png', 'rb') as f1,
    #       open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\DSC_0024.png', 'rb') as f2):
    #     images_data = [f1.read(), f2.read()]

    # result = train_classifier('user123')
    # print(result)
    test1()
