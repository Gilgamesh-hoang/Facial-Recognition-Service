import os
import pickle
import shutil
from collections import defaultdict
import io

import cv2
import numpy as np
from PIL import Image

def load_embedding(file_path):
    with open(file_path, 'rb') as f:
        embedding = pickle.load(f)
    return embedding.flatten()


def delete_empty_subdirectories(parent_dir):
    # Lặp qua tất cả các thư mục con trong thư mục cấp 1
    for dir_name in os.listdir(parent_dir):
        sub_dir_path = os.path.join(parent_dir, dir_name)

        # Kiểm tra xem có phải là thư mục không
        if os.path.isdir(sub_dir_path):
            # Đếm số lượng file trong thư mục cấp 2
            file_count = len([f for f in os.listdir(sub_dir_path) if os.path.isfile(os.path.join(sub_dir_path, f))])

            # Nếu số lượng file dưới n, xóa thư mục cấp 2
            if file_count <= 40:
                shutil.rmtree(sub_dir_path)
                print(f'Đã xóa thư mục: {sub_dir_path}')


def remove_empty_dirs(root_directory):
    # Duyệt qua từng thư mục trong thư mục gốc
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
        # Đếm số file trong thư mục
        file_count = len(filenames)

        # Nếu số file nhỏ hơn n, xóa thư mục
        if file_count < 3 and dirpath != root_directory:
            print(f"Đang xóa thư mục: {dirpath} (có {file_count} file)")
            shutil.rmtree(dirpath)


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


if __name__ == '__main__':
    path = 'D:\\Download\\lfw-funneled - Copy\\lfw_funneled'
    # remove_empty_dirs(path)
    # keep_max_files(path, 10)
    # count_files_in_subdirectories(path)
    # count_folder(path)

    pass
