import os
import pickle
from concurrent.futures.thread import ThreadPoolExecutor

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

import src.face_recognition.facenet as facenet
from src.align import detect_face

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACE_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'facemodel.pkl')
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
    with open('E:/Facial-Recognition-Service/Dataset/FaceData/embeddings.pkl', 'wb') as f:
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
        print(key, value[:2])
        i += 1
        if i == 2:
            break

def compare_models():
    with open('E:/Facial-Recognition-Service/Dataset/FaceData/embeddings.pkl', 'rb') as f:
        data = pickle.load(f)

    # Prepare data
    X = []  # Store all embedding vectors
    y = []  # Store corresponding labels (user IDs)

    # Iterate through each user and their corresponding embedding vectors
    for user_id, vectors in data.items():
        for vector in vectors:
            X.append(vector.flatten())  # Flatten the embeddings
            y.append(user_id)

    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # Split data into train/test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train SGDClassifier
    sgd_model = SGDClassifier(loss='log_loss', max_iter=1000, tol=0.001, random_state=42)
    sgd_model.fit(X_train, y_train)
    sgd_pred = sgd_model.predict(X_test)
    sgd_accuracy = accuracy_score(y_test, sgd_pred)

    # Train Naive Bayes
    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_pred = nb_model.predict(X_test)
    nb_accuracy = accuracy_score(y_test, nb_pred)

    # Present experimental results
    models = ['SGD', 'Naive Bayes']
    accuracies = [sgd_accuracy, nb_accuracy]

    plt.bar(models, accuracies, color=['blue', 'green'])
    plt.xlabel('Models')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of SGD vs. Naive Bayes')
    plt.ylim(0, 1)

    for i, acc in enumerate(accuracies):
        plt.text(i, acc + 0.02, f'{acc:.2f}', ha='center', fontsize=12, fontweight='bold')

    plt.show()

    print("Accuracy of SGDClassifier:", sgd_accuracy)
    print("Accuracy of Naive Bayes:", nb_accuracy)

if __name__ == '__main__':
    # write_embedding('D:\Download\lfw-funneled-Copy\lfw_funneled')
    # load_embedding('E:/Facial-Recognition-Service/Dataset/FaceData/embeddings.pkl')
    pass
