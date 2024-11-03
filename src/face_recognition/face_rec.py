import concurrent
import math
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import cv2
import tensorflow as tf
from src.align import detect_face
import concurrent.futures
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')  # Path to the FaceNet model

# Configuration parameters
MINSIZE = 20
THRESHOLD = [0.7, 0.7, 0.8]
FACTOR = 0.709
INPUT_IMAGE_SIZE = 160
IMAGE_PATH = 'E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\21130363.png'


def main():
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Load the MTCNN model for face detection
            print('Loading feature extraction model')
            # facenet.load_model(FACENET_MODEL_PATH)

            # Get input and output tensors
            # images_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("input:0")
            # embeddings = tf.compat.v1.get_default_graph().get_tensor_by_name("embeddings:0")
            # phase_train_placeholder = tf.compat.v1.get_default_graph().get_tensor_by_name("phase_train:0")
            # embedding_size = embeddings.get_shape()[1]

            # Initialize MTCNN networks
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            # Read the image
            frame = cv2.imread(IMAGE_PATH)

            # Detect faces in the image
            bounding_boxes, points = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]
            print(f'Number of faces found: {faces_found}')
            print('points:', points)
            # if faces_found > 0:
            #     det = bounding_boxes[:, 0:4]
            #     bb = np.zeros((faces_found, 4), dtype=np.int32)
            #     for i in range(faces_found):
            #         bb[i][0] = det[i][0]
            #         bb[i][1] = det[i][1]
            #         bb[i][2] = det[i][2]
            #         bb[i][3] = det[i][3]

            # Crop and display each face
            # cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
            # resized = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
            # prewhitened = facenet.prewhiten(resized)
            # reshaped = prewhitened.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)

            # Get the embedding vector
            # feed_dict = {images_placeholder: reshaped, phase_train_placeholder: False}
            # embedding = sess.run(embeddings, feed_dict=feed_dict)

            # Print the embedding vector
            # print(f'Embedding vector for face {i + 1}: {embedding}')

            # write the embedding vector to a file with path
            # with open('Dataset/FaceData/embedding3.pkl', 'wb') as f:
            #     pickle.dump(embedding, f)

            # Display the cropped face
            #         cv2.imshow(f'Face {i + 1}', cropped)
            #
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()


def detect_and_mark_landmarks(image_path):
    # Điểm 1: Trung tâm của mắt trái
    # Điểm 2: Trung tâm của mắt phải
    # Điểm 3: Trung tâm của mũi
    # Điểm 4: Góc trái của miệng
    # Điểm 5: Góc phải của miệng

    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            # Initialize MTCNN networks
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            # Read the image
            frame = cv2.imread(image_path)

            # Detect faces in the image
            bounding_boxes, points = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
            faces_found = bounding_boxes.shape[0]
            # print(f'Number of faces found: {faces_found}')
            # print('points:', points)
            # print('bounding_boxes:', bounding_boxes)

            # draw_landmarks(frame, points)

            file_name = os.path.basename(image_path)
            # save the image with landmarks to file with file name
            # cv2.imwrite(f'C:\\Users\\FPT SHOP\\Pictures\\landmarks_{file_name}', frame)

            # Display the image with landmarks
            # cv2.imshow(f'Image with Landmarks {file_name}', frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

            left_eye_x = int(points[0, 0])  # Tọa độ x mắt trái
            left_eye_y = int(points[5, 0])  # Tọa độ y mắt trái
            right_eye_x = int(points[1, 0])  # Tọa độ x mắt phải
            right_eye_y = int(points[6, 0])  # Tọa độ y mắt phải
            left_eye = (left_eye_x, left_eye_y)  # (x, y) của mắt trái
            right_eye = (right_eye_x, right_eye_y)  # (x, y) của mắt phải
            # print(f"Left eye: {left_eye}")
            # print(f"Right eye: {right_eye}")
            rotate_face_to_align_eyes(image_path, left_eye, right_eye)

def draw_landmarks(frame, points):
    # Duyệt qua từng cặp tọa độ của điểm landmark
    for i in range(points.shape[1]):  # Lặp qua các bức ảnh (N)
        # Lấy tọa độ x và y cho từng điểm
        left_eye_x = int(points[0, i])  # Tọa độ x mắt trái
        left_eye_y = int(points[5, i])  # Tọa độ y mắt trái
        right_eye_x = int(points[1, i])  # Tọa độ x mắt phải
        right_eye_y = int(points[6, i])  # Tọa độ y mắt phải
        nose_x = int(points[2, i])  # Tọa độ x mũi
        nose_y = int(points[7, i])  # Tọa độ y mũi
        mouth_left_x = int(points[3, i])  # Tọa độ x góc miệng trái
        mouth_left_y = int(points[8, i])  # Tọa độ y góc miệng trái
        mouth_right_x = int(points[4, i])  # Tọa độ x góc miệng phải
        mouth_right_y = int(points[9, i])  # Tọa độ y góc miệng phải

        # Vẽ hình tròn tại các điểm landmark
        cv2.circle(frame, (left_eye_x, left_eye_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (right_eye_x, right_eye_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (nose_x, nose_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (mouth_left_x, mouth_left_y), 2, (0, 255, 0), -1)
        cv2.circle(frame, (mouth_right_x, mouth_right_y), 2, (0, 255, 0), -1)

        # Đánh số cho từng điểm landmark
        cv2.putText(frame, f'{left_eye_x};{left_eye_y}', (left_eye_x + 5, left_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)  # Mắt trái
        cv2.putText(frame, f'{right_eye_x};{right_eye_y}', (right_eye_x + 5, right_eye_y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 0, 0), 1)  # Mắt phải
        cv2.putText(frame, f'{nose_x};{nose_y}', (nose_x + 5, nose_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0),
                    1)  # Mũi
        cv2.putText(frame, f'{mouth_left_x};{mouth_left_y}', (mouth_left_x + 5, mouth_left_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Miệng trái
        cv2.putText(frame, f'{mouth_right_x};{mouth_right_y}', (mouth_right_x + 5, mouth_right_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)  # Miệng phải


def plot_detected_faces(root_path):
    with tf.Graph().as_default():
        gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, os.path.join(BASE_DIR, 'src', 'align'))

            # Sử dụng ThreadPoolExecutor để xử lý ảnh song song
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = []

                for subdir, dirs, files in os.walk(root_path):
                    for file in files:
                        img_path = os.path.join(subdir, file)
                        # Thêm tác vụ xử lý ảnh vào thread pool
                        futures.append(executor.submit(process_image, img_path, pnet, rnet, onet))

                # Tạo một figure để hiển thị các ảnh xoay
                plt.figure(figsize=(10, len(futures)))
                names = []
                for future in concurrent.futures.as_completed(futures):
                    try:
                        img_path = future.result()  # Gọi result() để lấy kết quả hoặc ngoại lệ nếu có
                        if img_path is None:
                            continue
                        names.append(img_path)
                    except Exception as e:
                        print(f"Error processing image: {e}")
                names = sorted(names, key=lambda x: int(x.split('_')[-1].split('.')[0]))
                i = 0
                for name in names:
                    original_image = Image.open(name)
                    file_name = os.path.basename(name)
                    plt.subplot(len(futures), 5, i + 1)
                    plt.imshow(original_image)
                    plt.axis('off')
                    plt.title(f"{file_name}")
                    i += 1
                plt.tight_layout(h_pad=2)
                plt.show()


def process_image(img_path, pnet, rnet, onet):
    frame = cv2.imread(img_path)

    # Detect faces in the image
    bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)
    faces_found = bounding_boxes.shape[0]

    if faces_found > 0:
        # print(f'Number of faces found in {img_path}: {faces_found}')
        return img_path
    return None


def rotate_and_display_image(image_path):
    # Đọc ảnh từ đường dẫn
    original_image = Image.open(image_path)

    # Các góc xoay
    angles = [90, 180, 270]

    # Tạo một figure để hiển thị các ảnh xoay
    plt.figure(figsize=(10, 5))

    for i, angle in enumerate(angles):
        # Xoay ảnh
        rotated_image = original_image.rotate(angle)

        # Thêm ảnh xoay vào plot
        plt.subplot(1, len(angles), i + 1)
        plt.imshow(rotated_image)
        plt.axis('off')
        plt.title(f"{angle}°")

    # Hiển thị các ảnh xoay
    plt.show()


def rotate(image_path):
    # Đọc ảnh từ đường dẫn
    original_image = Image.open(image_path)

    # rotate image from 0 to 360 degrees with step 30 degrees
    # angles = range(0, 360, 5)
    angles = [30]

    for angle in angles:
        # Rotate the image
        rotated_image = original_image.rotate(angle)

        # write image to file
        rotated_image.save(f"E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated1_{angle}.png")


def rotate_face_to_align_eyes(image_path, left_eye, right_eye):
    image = Image.open(image_path)
    # Tọa độ của mắt trái và mắt phải
    (x1, y1) = left_eye
    (x2, y2) = right_eye

    # Tính góc xoay từ mắt trái đến mắt phải
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    print(f"Calculated Rotation Angle: {angle}")

    # Tính toán trung điểm của hai mắt, dùng làm điểm xoay
    eyes_center = ((x1 + x2) // 2, (y1 + y2) // 2)

    # Chuyển đổi ảnh từ PIL sang NumPy array
    image_np = np.array(image)

    # Lấy ma trận xoay ngược với góc tính được để làm ảnh nằm ngang
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, 1.0)

    # Xoay ảnh
    rotated_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]),
                                   flags=cv2.INTER_LINEAR)
    cv2.imwrite(f"E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated2.png", rotated_image)

    # get center of the image
    # cai nay dung de xoay anh theo tam cua anh neu co nhieu face
    # center = (image_np.shape[1] // 2, image_np.shape[0] // 2)
    # rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # rotated_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]),
    #                                flags=cv2.INTER_LINEAR)
    # cv2.imwrite(f"E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated3.png", rotated_image)

    # Lưu hoặc hiển thị ảnh đã xoay
    # cv2.imshow("Aligned Face", rotated_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # return rotated_image


# Sử dụng hàm với đường dẫn ảnh của bạn
# rotate("C:\\Users\\FPT SHOP\\Pictures\\Saved Pictures\\IMG_20240213_123347.jpg")
# detect_and_mark_landmarks("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_5.png")
detect_and_mark_landmarks("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_0.png")
detect_and_mark_landmarks("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_330.png")

