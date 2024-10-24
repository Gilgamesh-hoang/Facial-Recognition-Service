import os
import pickle

import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def train():
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\embeddings.pkl', 'rb') as f:
        data = pickle.load(f)

    # Step 2: Prepare features (X) and labels (y)
    X = []  # Embeddings
    y = []  # Corresponding labels

    for label, embeddings_list in data.items():
        for embedding in embeddings_list:
            X.append(embedding.flatten())  # Flatten the embeddings
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    # Step 3: Encode labels into numerical format
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Step 4: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Step 5: Set up the Naive Bayes model with GridSearchCV
    param_grid = {
        'var_smoothing': np.logspace(-12, -5, 20)  # Test 20 values from 1e-12 to 1e-5
    }

    # Initialize GaussianNB and GridSearchCV
    nb_model = GaussianNB()
    grid_search = GridSearchCV(estimator=nb_model, param_grid=param_grid,
                               scoring='accuracy', cv=5, n_jobs=-1, verbose=1)

    # Train the model with GridSearchCV
    grid_search.fit(X_train, y_train)

    # Step 6: Evaluate the best model on the test set
    best_nb_model = grid_search.best_estimator_
    y_pred = best_nb_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Test Set Accuracy: {accuracy:.4f}")

    # Print detailed classification report
    # print("Classification Report:\n", classification_report(y_test, y_pred))

    print("Training completed successfully.")
    print('best_nb_model', best_nb_model)

# best_nb_model GaussianNB(var_smoothing=1e-12)
train()
