import cv2
import numpy as np
import tensorflow as tf
import src.align.detect_face as detect_face
import facenet
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', help='Path to the image file.', required=True)
    args = parser.parse_args()

    # Configuration parameters
    MINSIZE = 20
    THRESHOLD = [0.7, 0.7, 0.8]
    FACTOR = 0.709
    INPUT_IMAGE_SIZE = 160
    IMAGE_PATH = args.path
    FACENET_MODEL_PATH = 'Models/20180402-114759.pb'

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

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
            pnet, rnet, onet = detect_face.create_mtcnn(sess, "src/align")

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

                    # Display the cropped face
                    cv2.imshow(f'Face {i + 1}', cropped)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

main()