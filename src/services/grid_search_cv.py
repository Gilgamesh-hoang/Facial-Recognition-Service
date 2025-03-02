import pickle

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder


# Best parameters SGD found: Best Parameters: {'alpha': 0.0001, 'learning_rate': 'optimal', 'loss': 'log_loss', 'max_iter': 1000, 'penalty': 'l2', 'random_state': 42}
def grid_search_cv_sgd():
    with open('/Dataset/embeddings.pkl', 'rb') as f:
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
        'loss': ['hinge', 'log_loss', 'modified_huber'],  # Các hàm mất mát khác nhau
        'penalty': ['l2', 'l1', 'elasticnet'],  # Các hình phạt regularization
        'alpha': [1e-4, 1e-3, 1e-2],  # Hệ số điều chỉnh regularization
        'learning_rate': ['constant', 'optimal', 'adaptive'],  # Kiểu cập nhật learning rate
        'max_iter': [1000, 2000, 3000]  # Số vòng lặp tối đa
    }

    # Initialize GaussianNB and GridSearchCV
    model = SGDClassifier()
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid,
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

    print("Training completed successfully.")
    print('best_nb_model', best_nb_model)
