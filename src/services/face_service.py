import concurrent
import os
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import tensorflow as tf
import concurrent.futures
import src.face_recognition.facenet as facenet
from src.align import detect_face
import src.utils.constant as constant


# sua lai ham process_image, lay all face trong 1 anh
def process_image(image_data, pnet, rnet, onet):
    frame = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
    bounding_boxes, _ = detect_face.detect_face(frame, constant.MINSIZE, pnet, rnet, onet, constant.THRESHOLD,
                                                constant.FACTOR)
    faces_found = bounding_boxes.shape[0]

    if faces_found == 1:
        bounding_box = bounding_boxes[0, 0:4].astype(int)
        cropped = frame[bounding_box[1]:bounding_box[3], bounding_box[0]:bounding_box[2], :]
        resized = cv2.resize(cropped, (constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE),
                             interpolation=cv2.INTER_CUBIC)
        prewhitened = facenet.prewhiten(resized)
        return prewhitened
    return None


def get_embeddings(images_data: list[bytes]) -> list[np.ndarray]:
    # Create a new TensorFlow graph and session once
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            facenet.load_model(constant.FACENET_MODEL_PATH)
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")

            pnet, rnet, onet = detect_face.create_mtcnn(sess, constant.DET_MODEL_DIR)

            # Tạo một thread pool
            embeddings_list = []
            frames = []

            # Sử dụng ThreadPoolExecutor để xử lý hình ảnh
            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                # Gửi các tác vụ vào thread pool
                futures = {executor.submit(process_image, imageData, pnet, rnet, onet): imageData for imageData in
                           images_data}

                # Chờ cho tất cả các thread hoàn thành và thu thập kết quả
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        frames.append(result)

            if frames:
                reshaped = np.stack(frames).reshape(-1, constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE, 3)
                feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                embeddings_list = sess.run(embeddings, feed_dict=feed_dict)

            sess.close()
            return embeddings_list


def rotate_image():
    pass


if __name__ == "__main__":
    # images = []
    # with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\21130363.png', 'rb') as file:
    #     data = file.read()
    #     images.append(data)
    #     images.append(data)
    #     images.append(data)
    # import time
    #
    # copy = images.copy()
    # start = time.time()
    # embeddings = get_embeddings(copy)
    # print(time.time() - start)

    # print('tensorflow:', tf.__version__)
    # print("Num GPUs Available: ", tf.test.is_gpu_available)
    # print(tf.test.gpu_device_name)
    # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #
    # import torch
    # print('torch:', torch.__version__)
    # print("Number of GPU: ", torch.cuda.device_count())
    # print("GPU Name: ", torch.cuda.get_device_name())
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device:', device)
    pass
