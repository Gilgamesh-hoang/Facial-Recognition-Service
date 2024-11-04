import pickle

import numpy as np
import tensorflow as tf

import src.face_recognition.facenet as facenet
import src.utils.constant as constant


def prewhiten_batch(images: list[np.ndarray]) -> np.ndarray:
    """Apply prewhiten to each image in the list and return as a single numpy array."""
    prewhitened_images = [facenet.prewhiten(img) for img in images]
    return np.stack(prewhitened_images)


def get_embeddings(images_data: list[np.ndarray]) -> list[np.ndarray]:
    # Tạo một TensorFlow graph và session duy nhất
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            facenet.load_model(constant.FACENET_MODEL_PATH)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            # Tiền xử lý toàn bộ images trong một lần
            prewhitened_images = prewhiten_batch(images_data)
            reshaped = prewhitened_images.reshape(-1, constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE, 3)

            # Chạy inference để tính toán embeddings
            feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
            embeddings_list = sess.run(embeddings, feed_dict=feed_dict)

        # Đóng session
        sess.close()

        return embeddings_list


if __name__ == "__main__":
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\list_ndarray.pkl', 'rb') as file:
        data = pickle.load(file)
    import time

    start = time.time()
    embeddings = get_embeddings(data)
    print(len(embeddings))
    print('time ms:', (time.time() - start) * 1000) #7655.388593673706 7151.320695877075 14.806.709289550781


    # print('tensorflow:', tf.__version__)
    # print("Num GPUs Available: ", tf.test.is_gpu_available)
    # print(tf.test.gpu_device_name)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

    # import torch
    # print('torch:', torch.__version__)
    # print("Number of GPU: ", torch.cuda.device_count())
    # print("GPU Name: ", torch.cuda.get_device_name())
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    # pass
