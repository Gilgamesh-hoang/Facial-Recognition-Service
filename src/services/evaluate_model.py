import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def load_embedding(file_path: str):
    """Load and return embeddings from a pickle file."""
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    #     print the first 2 rows of data
    # i = 0
    print(len(data))
    for key, value in data.items():
        # value = np.array(value)
        # print(value[0])
        print(key, len(value))
        # i += 1
        # if i == 10:
        #     break


def evaluate_model():
    # Load the data
    with open('/Dataset/FaceData/embeddings.pkl', 'rb') as f:
        (emb_array, labels) = pickle.load(f)


    X = np.array(emb_array, dtype=np.float64)
    y = np.array(labels)
    print(X.shape, y.shape)
    print(len(np.unique(y)))

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the model
    model_classify = SGDClassifier(alpha=0.0001, learning_rate='optimal', loss='log_loss', max_iter=1000, penalty='l2',
                                   random_state=42)
    model_classify.fit(X_train, y_train)

    # Print accuracy
    print("Accuracy: ", model_classify.score(X_test, y_test))

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    print("KNN Accuracy: ", knn.score(X_test, y_test))
    # Reinitialize and partially fit the model
    model_classify = SGDClassifier(alpha=0.0001, learning_rate='optimal', loss='log_loss', max_iter=1000, penalty='l2',
                                   random_state=42)
    model_classify.partial_fit(X_train, y_train, classes=np.unique(y))
    # Print accuracy
    print("Accuracy: ", model_classify.score(X_test, y_test))





if __name__ == '__main__':
    # load_embedding('E:/Facial-Recognition-Service/Dataset/FaceData/evaluate_embeddings.pkl')
    evaluate_model()