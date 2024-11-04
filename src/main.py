import json
import os
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from src.services.model_service import ModelService


class ImageRequest(BaseModel):
    imageURL: str


class ImagesRequest(BaseModel):
    user_id: str
    imageURLs: list[str]


# Khởi tạo service quản lý model
model_service = ModelService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan handler để load model khi ứng dụng khởi động."""
    print("Starting application...")
    model_service.load_components()
    yield  # Ứng dụng chạy tại đây
    print("Shutting down application...")
    model_service.save_components()


app = FastAPI(lifespan=lifespan)


# Dependency để inject service vào controller
def get_model_service() -> ModelService:
    return model_service


# Thêm middleware nếu cần (ví dụ: CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# @app.post("/identify-face")
# async def identify_face(request: ImageRequest, service: ModelService = Depends(get_model_service)):
#     imageURL = request.imageURL
#
#     if not imageURL:
#         raise HTTPException(status_code=400, detail="No image provided")
#
#     # Tải ảnh từ URL
#     try:
#         response = requests.get(imageURL)
#         response.raise_for_status()  # Kiểm tra nếu request thất bại
#         img_data = response.content  # Dữ liệu ảnh ở dạng bytes
#     except requests.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
#
#     # Gọi hàm nhận diện khuôn mặt với dữ liệu bytes của ảnh
#     response =  service.predict_user(img_data)
#     return response.to_dict()
@app.post("/identify-face")
async def identify_face(service: ModelService = Depends(get_model_service)):
    img_data = open("E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_350.png", "rb").read()
    # Gọi hàm nhận diện khuôn mặt với dữ liệu bytes của ảnh
    response = service.predict_user(img_data)
    return response.to_dict()

@app.post("/train-image")
async def upload_image_for_training(service: ModelService = Depends(get_model_service)):
    images = []
    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated_30.png',
              'rb') as file:
        data = file.read()
        images.append(data)

    with open('E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\hoang\\IMG_20230123_083911.png',
              'rb') as file:
        data = file.read()
        images.append(data)

    response = service.train_classifier('phi_hoang', images)
    return response.to_dict()


# @app.post("/train-image")
# async def upload_image_for_training(request: ImagesRequest, service: ModelService = Depends(get_model_service)):
#     if not request.user_id:
#         raise HTTPException(status_code=400, detail="No user_id provided")
#
#     imageURLs = request.imageURLs
#     if not imageURLs:
#         raise HTTPException(status_code=400, detail="No images provided")
#
#     images = []
#     try:
#         for url in imageURLs:
#             response = requests.get(url)
#             response.raise_for_status()
#             img_data = response.content
#             images.append(img_data)
#     except requests.RequestException as e:
#         raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
#
#     response = service.train_classifier(request.user_id, images)
#     return response.to_dict()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8111))  # Lấy port từ biến môi trường hoặc dùng 8000 mặc định
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
