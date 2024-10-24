from importlib import reload

import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import requests
from pydantic import BaseModel

from src.services import face_service

app = FastAPI()


class ImageRequest(BaseModel):
    imageURL: str
class ImagesRequest(BaseModel):
    imageURLs: list[str]


@app.post("/identify-face")
async def identify_face(request: ImageRequest):
    imageURL = request.imageURL

    if not imageURL:
        raise HTTPException(status_code=400, detail="No image provided")

    # Tải ảnh từ URL
    try:
        response = requests.get(imageURL)
        response.raise_for_status()  # Kiểm tra nếu request thất bại
        img_data = response.content  # Dữ liệu ảnh ở dạng bytes
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Error fetching image: {str(e)}")

    # Gọi hàm nhận diện khuôn mặt với dữ liệu bytes của ảnh
    face_service.identify_face(img_data)

    return {"message": "Face identification completed"}

@app.post("/extract-vectors")
async def extract_vectors(request: ImagesRequest):
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

    face_service.extract_face_vectors(images)

    return {"message": "Face vectors extracted"}


# image = np.frombuffer(await file.read(), np.uint8)
# input_vector = extract_face_vector(image)
# if input_vector is None:
#     raise HTTPException(status_code=400, detail="No face detected")
#
# user = db.find_one({"face_vector": {"$near": input_vector}})
# if user:
#     return {"user_id": str(user['_id'])}
# else:
#     raise HTTPException(status_code=404, detail="User not found")
# @app.get("/")
# async def root():
#     return {"message": "Hello World"}
#
#
# @app.get("/hello/{name}")
# async def say_hello(name: str):
#     return {"message": f"Hello {name}"}

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8111))  # Lấy port từ biến môi trường hoặc dùng 8000 mặc định
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
