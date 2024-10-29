import os
from contextlib import asynccontextmanager

import requests
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
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
    model_service.load_model_from_file()
    yield  # Ứng dụng chạy tại đây
    print("Shutting down application...")
    model_service.save_model_to_file()

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

@app.post("/identify-face")
async def identify_face(request: ImageRequest):
    # imageURL = request.imageURL
    #
    # if not imageURL:
    #     raise HTTPException(status_code=400, detail="No image provided")
    #
    # # Tải ảnh từ URL
    # try:
    #     response = requests.get(imageURL)
    #     response.raise_for_status()  # Kiểm tra nếu request thất bại
    #     img_data = response.content  # Dữ liệu ảnh ở dạng bytes
    # except requests.RequestException as e:
    #     raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")
    #
    # # Gọi hàm nhận diện khuôn mặt với dữ liệu bytes của ảnh
    # face_service.identify_face(img_data)
    #
    # return {"message": "Face identification completed"}
    pass

@app.post("/train-image")
async def upload_image_for_training(request: ImagesRequest , service: ModelService = Depends(get_model_service)):
    if not request.user_id:
        raise HTTPException(status_code=400, detail="No user_id provided")

    imageURLs = request.imageURLs
    if not imageURLs:
        raise HTTPException(status_code=400, detail="No images provided")

    images = []
    try:
        for url in imageURLs:
            response = requests.get(url)
            response.raise_for_status()
            img_data = response.content
            images.append(img_data)
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

    service.train_classifier(request.user_id, images)

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8111))  # Lấy port từ biến môi trường hoặc dùng 8000 mặc định
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
