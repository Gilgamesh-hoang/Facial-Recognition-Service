import os

import cv2
import numpy as np
import tensorflow as tf

import src.face_recognition.facenet as facenet
from src.align import detect_face

# Get the base directory of the project (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'face-model.pkl')
FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model

def get_embeddings(images_data: list[bytes]) -> list[np.ndarray]:
    # Configuration parameters for face detection and recognition
    MINSIZE = 20  # Minimum size of the face
    THRESHOLD = [0.7, 0.7, 0.8]  # Three steps' threshold
    FACTOR = 0.709  # Scale factor
    INPUT_IMAGE_SIZE = 160  # Size of the input image for the model

    # Create a new TensorFlow graph and session once
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            facenet.load_model(FACENET_MODEL_PATH)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            embeddings_list = []
            frames = []
            for imageData in images_data:
                frame = cv2.imdecode(np.frombuffer(imageData, np.uint8), cv2.IMREAD_COLOR)
                bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
                faces_found = bounding_boxes.shape[0]
                if faces_found == 1:
                    bounding_box = bounding_boxes[0, 0:4].astype(int)
                    cropped = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
                    resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    prewhitened = facenet.prewhiten(resized)
                    frames.append(prewhitened)

            if frames:
                reshaped = np.stack(frames).reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                embeddings_list = sess.run(embeddings, feed_dict=feed_dict)

            sess.close()
            return embeddings_list

if __name__ == "__main__":
    images = []
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\21130363.png', 'rb') as file:
        data = file.read()
        images.append(data)
        images.append(data)
        images.append(data)
