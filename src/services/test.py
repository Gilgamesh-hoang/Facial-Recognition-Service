import pickle

import cv2
import numpy as np

import src.services.classification as classification


def test1():
    # label = classification.load_label_encode_from_file()
    # print(len(label.classes_))
#     print every row is 10 elements
#     for i in range(0, len(label.classes_), 10):
#         print(label.classes_[i:i+10])
#     print(label.inverse_transform([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]))

    with open('/Dataset/FaceData/embeddings.pkl', 'rb') as f:
        (data, label) = pickle.load(f)
        print(len(data))
        print(len(label))

    print(label[:20])




if __name__ == '__main__':
    test1()