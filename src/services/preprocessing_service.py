import concurrent
import math

import numpy as np
from sklearn.metrics import pairwise_distances
import cv2
import tensorflow as tf
from src.align.detect_face import detect_face, create_mtcnn
import concurrent.futures
import src.utils.constant as constant


class PreprocessingService:
    def __init__(self, face_number_per_img: int = 1):
        self.__pnet, self.__rnet, self.__onet = self.get_mtcnn()
        self.__face_number_per_img = face_number_per_img

    def get_mtcnn(self):
        with tf.Graph().as_default():
            gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
            sess = tf.compat.v1.Session(
                config=tf.compat.v1.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
            )
            with sess.as_default():
                pnet, rnet, onet = create_mtcnn(sess, constant.DET_MODEL_DIR)
                return pnet, rnet, onet

    def pre_process_image(self, images_data: list[np.ndarray]) -> list[np.ndarray]:
        # Sử dụng ThreadPoolExecutor để xử lý hình ảnh
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # Gửi các tác vụ vào thread pool
            futures = []
            for image in images_data:
                frame = cv2.imdecode(image, cv2.IMREAD_COLOR)
                frame = resize_image(frame, 800, 800)
                futures.append(executor.submit(self.process_image, frame, 90))

            # Chờ cho tất cả các thread hoàn thành và thu thập kết quả
            frames = []
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                if result is not None:
                    frames.extend(result)

        return frames

    def process_image(self, frame: np.ndarray, angle=90) -> list[np.ndarray] | None:
        # frame = cv2.imdecode(image_data, cv2.IMREAD_COLOR)
        bounding_boxes, points = detect_face(frame, constant.MINSIZE, self.__pnet, self.__rnet, self.__onet,
                                             constant.THRESHOLD, constant.FACTOR)
        faces_found = bounding_boxes.shape[0]
        if faces_found > self.__face_number_per_img:
            return None

        if faces_found > 0:
            left_eye, right_eye, _, _, _ = self.get_coordinates(points)
            frame = self.rotate_face_to_align_eyes(frame, left_eye, right_eye, faces_found != 1)
            return self.crop_and_resize(frame)

        if angle >= 270:
            return None
        # rotate image
        rotated_image = cv2.rotate(frame, angle)
        # get bytes of rotated image
        rotated_image_bytes = cv2.imencode('.jpg', rotated_image)[1].tobytes()
        return self.process_image(np.frombuffer(rotated_image_bytes, np.uint8), angle + 90)

    def crop_and_resize(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Detects faces in the frame, crops and resizes each detected face to a specific size.

        Parameters:
            frame (np.ndarray): Input image frame from which faces will be detected and cropped.

        Returns:
            list[np.ndarray]: A list of cropped and resized face images.
        """
        # Detect faces in the input frame
        bounding_boxes, _ = detect_face(frame, constant.MINSIZE, self.__pnet, self.__rnet, self.__onet,
                                        constant.THRESHOLD,
                                        constant.FACTOR)
        num_faces = bounding_boxes.shape[0]

        cropped_faces = []
        for i in range(num_faces):
            # Get bounding box for each face and round to integers
            x1, y1, x2, y2 = bounding_boxes[i, :4].astype(int)

            # Ensure bounding box coordinates are within frame boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)

            # Crop the face region
            cropped_face = frame[y1:y2, x1:x2, :]

            # Resize the cropped face
            resized_face = cv2.resize(cropped_face, (constant.INPUT_IMAGE_SIZE, constant.INPUT_IMAGE_SIZE),
                                      interpolation=cv2.INTER_CUBIC)

            # Append to results list
            cropped_faces.append(resized_face)

        return cropped_faces

    def get_coordinates(self, points):
        left_eye_x = int(points[0, 0])
        left_eye_y = int(points[5, 0])
        right_eye_x = int(points[1, 0])
        right_eye_y = int(points[6, 0])
        nose_x = int(points[2, 0])
        nose_y = int(points[7, 0])
        mouth_left_x = int(points[3, 0])
        mouth_left_y = int(points[8, 0])
        mouth_right_x = int(points[4, 0])
        mouth_right_y = int(points[9, 0])

        left_eye = (left_eye_x, left_eye_y)
        right_eye = (right_eye_x, right_eye_y)
        nose = (nose_x, nose_y)
        mouth_left = (mouth_left_x, mouth_left_y)
        mouth_right = (mouth_right_x, mouth_right_y)
        return left_eye, right_eye, nose, mouth_left, mouth_right

    def rotate_face_to_align_eyes(self, image: np.ndarray, left_eye: tuple[int, int], right_eye: tuple[int, int],
                                  is_only_face: bool = True) -> np.ndarray:
        """
        Xoay ảnh để căn chỉnh khuôn mặt dựa trên tọa độ của mắt.

        Parameters:
            image (np.ndarray): Ảnh gốc dưới dạng mảng NumPy.
            left_eye (tuple[int, int]): Tọa độ của mắt trái.
            right_eye (tuple[int, int]): Tọa độ của mắt phải.
            is_only_face (bool): Nếu `True`, xoay ảnh dựa trên trung điểm của hai mắt; nếu `False`, dùng trung điểm của ảnh.

        Returns:
            np.ndarray: Ảnh đã được xoay.
        """
        # image = Image.open(image_path)
        # Tọa độ của mắt trái và mắt phải
        (x1, y1) = left_eye
        (x2, y2) = right_eye

        # Tính góc xoay từ mắt trái đến mắt phải
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))

        if abs(angle) < 5:
            return image

        # Chuyển đổi ảnh từ PIL sang NumPy array
        image_np = np.array(image)

        if is_only_face:
            # Tính toán trung điểm của hai mắt, dùng làm điểm xoay
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
        else:
            # Tính toán trung điểm của ảnh, dùng làm điểm xoay
            center = (image_np.shape[1] // 2, image_np.shape[0] // 2)

        # Lấy ma trận xoay ngược với góc tính được để làm ảnh nằm ngang
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Xoay ảnh
        rotated_image = cv2.warpAffine(image_np, rotation_matrix, (image_np.shape[1], image_np.shape[0]),
                                       flags=cv2.INTER_LINEAR)
        return rotated_image


def resize_image(image: np.ndarray, max_width: int, max_height: int) -> np.ndarray:
    if image is None or len(image.shape) < 2:
        raise ValueError("Invalid image provided")
    # Bước 1: Lấy kích thước hiện tại của ảnh
    height, width = image.shape[:2]

    # Bước 2: Kiểm tra nếu ảnh vượt quá kích thước tối đa
    if width > max_width or height > max_height:
        # Tính toán tỉ lệ để giữ nguyên tỷ lệ của ảnh khi resize
        scaling_factor = min(max_width / width, max_height / height)
        new_width = int(width * scaling_factor)
        new_height = int(height * scaling_factor)
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        return image


def remove_outliers(embeddings: list[np.ndarray], threshold=0.4) -> list[np.ndarray] | None:
    # Calculate Cosine distances between all vectors
    num_vectors = len(embeddings)
    distances = pairwise_distances(embeddings, metric='cosine')
    distances = np.round(distances, 4)
    rows_without_diagonal = [np.delete(distances[i], i) for i in range(distances.shape[0])]
    # Danh sách lưu trữ các chỉ số của outliers
    outliers_indices = []

    for i, row in enumerate(rows_without_diagonal):
        if np.all(row > threshold):
            outliers_indices.append(i)
        elif num_vectors > 3 and np.sum(row < threshold) == 1:
            # Nếu có nhieu hon 3 vector và chỉ có 1 vector có khoảng cách nhỏ hơn ngưỡng
            # thì vector đó cũng được coi là outlier
            outliers_indices.append(i)

    # Xử lý các trường hợp đặc biệt
    num_outliers = len(outliers_indices)
    print('outliers_indices: ', outliers_indices)

    if num_outliers == 0:
        # Không có outlier
        return embeddings
    elif num_outliers < num_vectors / 2:
        # Outlier chiếm phần nhỏ, loại bỏ chúng
        return [embedding for i, embedding in enumerate(embeddings) if i not in outliers_indices]
    else:
        # Có nhiều outlier hoặc mỗi vector là một khuôn mặt khác nhau, trả về danh sách ban đầu
        return None
