import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

# Get the base directory of the project (parent of src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DET_MODEL_DIR = os.path.join(BASE_DIR, 'src', 'align')

FACENET_MODEL_PATH = os.path.join(BASE_DIR, 'Models', '20180402-114759.pb')

CLASSIFY_MODEL_PATH = os.path.join(BASE_DIR, 'Models', 'face-model.pkl')

LABEL_ENCODE_PATH = os.path.join(BASE_DIR, 'Models', 'label-encode.pkl')

DATASET_EMBEDDINGS_PATH = os.path.join(BASE_DIR, 'Dataset', 'embeddings.pkl')

MINSIZE = 20  # Minimum size of the face

THRESHOLD = [0.7, 0.7, 0.8]  # Three steps' threshold

FACTOR = 0.709  # Scale factor

INPUT_IMAGE_SIZE = 160  # Size of the input image for the model
