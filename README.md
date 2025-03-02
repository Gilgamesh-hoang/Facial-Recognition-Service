<div align="center" style="position: relative;">
<h1>Facial Recognition Service</h1>
</div>

##  Table of Contents

- [Overview](#overview)
- [Technologies Used](#technologies-used)
- [Features](#features)
- [ Getting Started](#getting-started)
  - [ Prerequisites](#prerequisites)
  - [ Installation](#installation)
- [Usage](#usage)
- [ License](#-license)

---

##  Overview

This project is a facial recognition service that uses machine learning models to identify and classify faces in images. The service includes preprocessing, face detection, and classification components.

---

## Technologies Used

- **FastAPI:** A modern, fast (high-performance) web framework for building APIs with Python.
- **MTCNN:** A deep learning-based face detector known for its accuracy and speed.
- **FaceNet:** A deep learning model that produces facial embeddings for recognition tasks.
- **SGD Classification:** Stochastic Gradient Descent (SGD) is a popular optimization algorithm used in machine learning.
---

##  Features

- **Face Detection:** Utilizes MTCNN (Multi-task Cascaded Convolutional Neural Network) for accurate face detection.
- **Face Recognition:** Employs FaceNet to extract facial embeddings for recognition tasks.
- **API Endpoints:** Provides RESTful API endpoints for face detection and recognition using FastAPI.
- **SGD Classification:** Stochastic Gradient Descent (SGD) is used to train and predict the classification of facial embeddings.

---

##  Getting Started

###  Prerequisites

Before getting started with src, ensure your runtime environment meets the following requirements:

- **Python 3.12**: Ensure you have Python installed. You can download it from the [official Python website](https://www.python.org/).
- **pip**: Python package installer. It usually comes with Python installations.

###  Installation

Install src using one of the following methods:

**Build from source:**

1. Clone the src repository:
```sh
❯ git clone https://github.com/Gilgamesh-hoang/Facial-Recognition-Service.git
```

2. Navigate to the project directory:
```sh
❯ cd Facial-Recognition-Service
```

3. Create and activate a virtual environment:
```sh
❯ python -m venv venv
❯ source venv/bin/activate # On Windows: env\Scripts\activate
```

4. Install the project dependencies:
```sh
❯ pip install -r requirements.txt
```

5. Create a `Models` folder in the root directory and download the FaceNet model from [here](https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view) and place it in the `Models` folder.

6. Create a `Dataset` folder in the root directory and download the embedding dataset from [here](https://drive.google.com/file/d/1YqUYF3V0-LKzNqgjeBXaCPEiYlf-MGiH/view?usp=sharing) and place it in the `Dataset` folder.

---

## Usage
1. Run the FastAPI server:
```sh
❯ python src/main.py
```

2. Access the API documentation: Open your browser and navigate to http://localhost:8111/docs to view the interactive API documentation provided by Swagger UI.


---

##  License

This project is protected under the MIT License. For more details, refer to the [LICENSE](https://choosealicense.com/licenses/) file.

---
