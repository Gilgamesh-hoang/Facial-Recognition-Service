import math
import os
import pickle
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["TF_NUM_INTRAOP_THREADS"] = "2"

import cv2
import numpy as np
import tensorflow as tf

import src.face_recognition.facenet as facenet
from src.align import detect_face
import src.utils.constant as constant


def generate_embedding(input_paths, output_path):
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load the MTCNN model for face detection
            print('Loading feature extraction model')
            facenet.load_model(constant.FACENET_MODEL_PATH)

            # Get input and output tensors
            images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            # Initialize MTCNN networks
            pnet, rnet, onet = detect_face.create_mtcnn(sess, constant.DET_MODEL_DIR)

            for input_path in input_paths:
                # Read the image
                frame = cv2.imread(input_path)

                # Detect faces in the image
                bounding_boxes, _ = detect_face.detect_face(frame, constant.MINSIZE, pnet, rnet, onet,
                                                            constant.THRESHOLD,
                                                            constant.FACTOR)
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
                        resized = cv2.resize(cropped, (constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE),
                                             interpolation=cv2.INTER_CUBIC)
                        prewhitened = facenet.prewhiten(resized)
                        reshaped = prewhitened.reshape(-1, constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE, 3)

                        # Get the embedding vector
                        feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
                        embedding = sess.run(embeddings, feed_dict=feed_dict)

                        # Print the embedding vector
                        # print(f'Embedding vector for face {i + 1}: {embedding}')

                        # write the embedding vector to a file with path
                        # get file name without extension
                        filename = os.path.basename(input_path).split('.')[0] + '.pkl'
                        output = os.path.join(output_path, filename)
                        with open(output, 'wb') as f:
                            pickle.dump(embedding, f)

                else:
                    print('No faces found in the image.')


if __name__ == '__main__':
    # write_embedding('D:\\Download\\lfw-funneled - Copy\\lfw_funneled')
    # generate_embedding([
    #     "E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\IMG_20230126_123256.png",
    # ],
    #     'E:\\Facial-Recognition-Service\\Dataset\\FaceData')
    pass
