import os

import cv2
import numpy as np
import tensorflow as tf
from user_service import get_all_users

import src.face_recognition.facenet as facenet
from src.align import detect_face

# Get the base directory of the project (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def identify_face(imageData: bytes):
    # Configuration parameters for face detection and recognition
    MINSIZE = 20  # Minimum size of the face
    THRESHOLD = [0.7, 0.7, 0.8]  # Three steps' threshold
    FACTOR = 0.709  # Scale factor
    INPUT_IMAGE_SIZE = 160  # Size of the input image for the model
    FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model

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

            # Decode the image bytes into an OpenCV frame
            frame = cv2.imdecode(np.frombuffer(imageData, np.uint8), cv2.IMREAD_COLOR)
            # Detect faces in the image
            bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]  # Number of faces found
            if faces_found == 0:
                return {
                    "status": "error",
                    "message": "No faces found in the image"
                }
            elif faces_found > 1:
                return {
                    "status": "error",
                    "message": "Multiple faces found in the image"
                }
            # Initialize an array to hold bounding box coordinates
            bounding_box_coordinates = bounding_boxes[:, 0:4]
            bounding_boxes_array = np.zeros((faces_found, 4), dtype=np.int32)
            for i in range(faces_found):
                # Extract bounding box coordinates for each face
                bounding_boxes_array[i][0] = bounding_box_coordinates[i][0]
                bounding_boxes_array[i][1] = bounding_box_coordinates[i][1]
                bounding_boxes_array[i][2] = bounding_box_coordinates[i][2]
                bounding_boxes_array[i][3] = bounding_box_coordinates[i][3]

                # Crop and resize each face
                cropped = frame[bounding_boxes_array[i][1]:bounding_boxes_array[i][3], bounding_boxes_array[i][0]:bounding_boxes_array[i][2], :]
                resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                prewhitened = facenet.prewhiten(resized)  # Prewhiten the image
                reshaped = prewhitened.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)  # Reshape for the model

                # Get the embedding vector for the face
                feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                embedding = sess.run(embeddings, feed_dict=feed_dict)  # Run the session to get the embedding

                all_users = get_all_users()
                list_user_id = []
                for user in all_users:
                    if compare_embeddings(user.face_embedded_vector, embedding):
                        list_user_id.append(user.user_id)

            return {
                "status": "success",
                "user_ids": list_user_id
            }
                
def compare_embeddings(embedding1, embedding2, threshold= 0.8):
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    if distance < threshold:
        return True
    else:
        return False


def extract_face_vectors(images: list[bytes]):
    return None


if __name__ == "__main__":
    # Test the face identification function
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\raw\\hoang\\IMG_20240213_123743.jpg', 'rb') as file:
        identify_face(file.read())

