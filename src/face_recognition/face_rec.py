import concurrent
import os
import sys

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

            draw_landmarks(frame, points)

            file_name = os.path.basename(image_path)
            # save the image with landmarks to file with file name
            # cv2.imwrite(f'C:\\Users\\FPT SHOP\\Pictures\\landmarks_{file_name}', frame)

            # Display the image with landmarks
            cv2.imshow(f'Image with Landmarks {file_name}', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def draw_landmarks(frame, points):
    for i in range(points.shape[1]):
        for j in range(5):
            # Tính toán tọa độ của điểm landmark
            x = int(points[j, i])
            y = int(points[j + 5, i])

            # Vẽ hình tròn tại điểm landmark
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            # Đánh số cho từng điểm landmark
            text = str(j + 1)  # Tạo chuỗi số từ 1 đến 5
            cv2.putText(frame, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)


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
    angles = range(0, 360, 5)

    for angle in angles:
        # Rotate the image
        rotated_image = original_image.rotate(angle)

        # write image to file
        rotated_image.save(f"E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_{angle}.png")


# Sử dụng hàm với đường dẫn ảnh của bạn
# rotate("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\raw\\hoang\\21130363.jpg")
# main2("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate")
detect_and_mark_landmarks("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_325.png")
detect_and_mark_landmarks("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_110.png")

