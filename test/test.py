import logging
import pickle
import numpy as np
from collections import defaultdict
import os
import shutil
import requests
from PIL import Image
from io import BytesIO

def load_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding


def compare_embeddings(embedding1, embedding2, threshold=0.8):
    distance = np.linalg.norm(np.array(embedding1) - np.array(embedding2))
    print(f'Euclidean distance between embeddings: {distance}')
    if distance < threshold:
        print('The faces are of the same person.')
    else:
        print('The faces are of different persons.')


def compare():
    # Load embeddings from .pkl files
    embedding1 = load_embedding('/Dataset/FaceData/embedding2.pkl')
    embedding2 = load_embedding('/Dataset/FaceData/embedding3.pkl')

    # Compare embeddings
    compare_embeddings(embedding1, embedding2)


def display_image_from_url():
    url = "https://i1-dulich.vnecdn.net/2021/07/16/1-1626437591.jpg?w=0&h=0&q=100&dpr=2&fit=crop&s=yCCOAE_oJHG0iGnTDNgAEA"
    # Download image data from the URL
    response = requests.get(url)
    response.raise_for_status()  # Ensure the request was successful

    # print bytes size of the image
    print(f'Image size: {len(response.content)} bytes')

    # Convert the binary data to an image
    img = Image.open(BytesIO(response.content))

    # Display the image
    img.show()


# display_image_from_url()

def delete_empty_subdirectories(parent_dir):
    # Lặp qua tất cả các thư mục con trong thư mục cấp 1
    for dir_name in os.listdir(parent_dir):
        sub_dir_path = os.path.join(parent_dir, dir_name)

        # Kiểm tra xem có phải là thư mục không
        if os.path.isdir(sub_dir_path):
            # Đếm số lượng file trong thư mục cấp 2
            file_count = len([f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))])

            # Nếu số lượng file dưới 3, xóa thư mục cấp 2
            if file_count <= 40:
                shutil.rmtree(sub_dir_path)
                print(f'Đã xóa thư mục: {sub_dir_path}')


# Gọi hàm với đường dẫn thư mục cấp 1
# parent_directory = 'D:\Download\lfw-funneled-Copy\lfw_funneled'  # Thay đổi đường dẫn này
# delete_empty_subdirectories(parent_directory)
def remove_empty_dirs(root_directory):
    # Duyệt qua từng thư mục trong thư mục gốc
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
        # Đếm số file trong thư mục
        file_count = len(filenames)

        # Nếu số file nhỏ hơn n, xóa thư mục
        if file_count < 4 and dirpath != root_directory:
            print(f"Đang xóa thư mục: {dirpath} (có {file_count} file)")
            shutil.rmtree(dirpath)


# remove_empty_dirs('D:\Download\lfw-funneled-Copy\lfw_funneled')
def keep_max_files(root_directory, max_files=20):
    # Duyệt qua từng thư mục trong thư mục gốc
    for dirpath, dirnames, filenames in os.walk(root_directory):
        # Kiểm tra nếu số file lớn hơn max_files
        if len(filenames) > max_files:
            print(f"Thư mục {dirpath} có {len(filenames)} file, giữ lại {max_files} file.")

            # Lấy danh sách các file và sắp xếp theo tên
            files_to_keep = sorted(filenames)[:max_files]
            files_to_keep_set = set(files_to_keep)  # Chuyển danh sách giữ lại thành set để kiểm tra

            # Duyệt qua các file và xóa các file không nằm trong danh sách giữ lại
            for filename in filenames:
                if filename not in files_to_keep_set:
                    file_path = os.path.join(dirpath, filename)
                    print(f"Đang xóa file: {file_path}")
                    os.remove(file_path)  # Xóa file


# keep_max_files('D:\Download\lfw-funneled-Copy2\lfw_funneled', max_files=7)
def count_files_in_subdirectories(parent_dir):
    file_count_dict = defaultdict(int)

    # Lặp qua tất cả các thư mục con trong thư mục cấp 1
    for dir_name in os.listdir(parent_dir):
        sub_dir_path = os.path.join(parent_dir, dir_name)

        # Kiểm tra xem có phải là thư mục không
        if os.path.isdir(sub_dir_path):
            # Đếm số lượng file trong thư mục cấp 2
            file_count = len([f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))])
            file_count_dict[dir_name] = file_count

    # Sắp xếp theo số lượng file giảm dần
    sorted_file_counts = sorted(file_count_dict.items(), key=lambda x: x[1], reverse=True)

    # In ra kết quả
    print("Số lượng file trong các thư mục cấp 2:")
    for dir_name, count in sorted_file_counts:
        print(f'{dir_name}: {count} file(s)')


def count_folder(parent_dir):
    count = 0
    for dir_name in os.listdir(parent_dir):
        sub_dir_path = os.path.join(parent_dir, dir_name)
        if os.path.isdir(sub_dir_path):
            count += 1
    print(count)


# count_folder('D:\Download\lfw-funneled-Copy\lfw_funneled')
# print('=====================')
# count_folder('D:\Download\lfw-funneled-Copy2\lfw_funneled')

# Gọi hàm với đường dẫn thư mục cấp 1
# parent_directory = 'D:\Download\lfw-funneled-Copy\lfw_funneled'  # Thay đổi đường dẫn này
# count_files_in_subdirectories(parent_directoryđề
