import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import numpy as np
import tensorflow as tf

import src.face_recognition.facenet as facenet
from src.align import detect_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'face-model.pkl')
FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model

# Global variables to reuse model and session
MINSIZE = 20  # Minimum size of the face
THRESHOLD = [0.7, 0.7, 0.8]  # MTCNN thresholds
FACTOR = 0.709  # Scale factor
INPUT_IMAGE_SIZE = 160  # Input size for FaceNet


def initialize_session():
    """Initialize and return the TensorFlow session and FaceNet model."""
    tf.compat.v1.disable_eager_execution()  # Disable eager execution
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

    with sess.as_default():
        facenet.load_model(FACENET_MODEL_PATH)
        images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
        pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

    return sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet


def process_image(file_path, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet):
    """Process a single image and return its embedding vector."""
    frame = cv2.imread(file_path)
    if frame is None:
        print(f"Unable to read {file_path}. Skipping.")
        return None

    # Detect faces
    bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    if bounding_boxes.shape[0] != 1:  # Skip if not exactly one face detected
        return None
    # Extract and preprocess the face
    x1, y1, x2, y2 = bounding_boxes[0][:4].astype(int)
    cropped = frame[y1:y2, x1:x2, :]
    if cropped.size == 0:  # Check if the cropped image is empty
        print(f"Empty cropped image for {file_path}. Skipping.")
        return None

    resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    prewhitened = facenet.prewhiten(resized).reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

    # Compute the embedding
    feed_dict = {images_placeholder: prewhitened, phase_train_placeholder: False}
    embedding = sess.run(embeddings, feed_dict=feed_dict)
    return embedding


def process_user_folder(sub_dir_path, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet):
    """Process all images in a user's folder and return a list of embeddings."""
    print(f"Processing {sub_dir_path}")

    embeddings_list = []
    for file_name in os.listdir(sub_dir_path):
        file_path = os.path.join(sub_dir_path, file_name)
        embedding = process_image(file_path, sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet,
                                  onet)
        if embedding is not None:
            embeddings_list.append(embedding)
    return embeddings_list


def write_embedding(path: str):
    """Main function to extract and store embeddings."""
    sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet = initialize_session()

    results = {}
    with ThreadPoolExecutor() as executor:  # Use thread pool for parallel processing
        futures = {
            executor.submit(process_user_folder, os.path.join(path, dir_name),
                            sess, images_placeholder, embeddings, phase_train_placeholder, pnet, rnet, onet): dir_name
            for dir_name in os.listdir(path) if os.path.isdir(os.path.join(path, dir_name))
        }

        # Collect results as they complete
        for future in futures:
            dir_name = futures[future]
            embeddings_list = future.result()
            if embeddings_list:
                results[dir_name] = embeddings_list

    # Save results to a pickle file
    with open('E:/Facial-Recognition-Service/Dataset/FaceData/hoang_embeddings.pkl', 'wb') as f:
        pickle.dump(results, f)

    print("Embeddings extraction completed successfully.")
    sess.close()


def load_embedding(file_path: str):
    """Load and return embeddings from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    #     print the first 2 rows of data
    i = 0
    for key, value in data.items():
        value = np.array(value)
        print(key, len(value))
        i += 1
        if i == 20:
            break


def generate_embedding():
    IMAGE_PATH = "E:/Facial-Recognition-Service/Dataset/FaceData/processed/hoang/IMG_20240213_123347.jpg"

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load the MTCNN model for face detection
            print('Loading feature extraction model')
            facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Initialize MTCNN networks
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            # Read the image
            frame = cv2.imread(IMAGE_PATH)

            # Detect faces in the image
            bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]

            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # Crop and display each face
                    cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                    resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                    prewhitened = facenet.prewhiten(resized)
                    reshaped = prewhitened.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

                    # Get the embedding vector
                    feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                    embedding = sess.run(embeddings, feed_dict=feed_dict)

                    # Print the embedding vector
                    print(f'Embedding vector for face {i + 1}: {embedding}')

                    # write the embedding vector to a file with path
                    with open('E:/Facial-Recognition-Service/Dataset/FaceData/embedding2.pkl', 'wb') as f:
                        pickle.dump(embedding, f)

            else:
                print('No faces found in the image.')


if __name__ == '__main__':
    # write_embedding('E:\Facial-Recognition-Service\Dataset\FaceData\processed')
    # load_embedding('E:\Facial-Recognition-Service\Dataset\FaceData\hoang_embeddings.pkl')
    # compare_models()
    generate_embedding()
