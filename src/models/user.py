import uuid
import numpy as np
from bson import Binary

class User:
    def __init__(self, user_id: uuid.UUID, face_embedded_vector: np.ndarray):
        self.user_id = user_id
        self.face_embedded_vector = face_embedded_vector

    def to_dict(self) -> dict:
        """Chuyển đối tượng User thành dictionary để lưu vào MongoDB."""
        return {
            "_id": Binary.from_uuid(self.user_id),  # Lưu UUID dưới dạng binary
            "face_embedded_vector": Binary(self.face_embedded_vector.tobytes())  # Chuyển vector thành binary
        }

    @staticmethod
    def from_dict(data: dict) -> "User":
        """Tạo đối tượng User từ dữ liệu trong MongoDB."""
        vector = np.frombuffer(data["face_embedded_vector"], dtype=np.float64)
        user_id = data["_id"]
        if isinstance(user_id, Binary):
            user_id = uuid.UUID(bytes=user_id)
        return User(
            user_id=user_id,
            face_embedded_vector=vector
        )

    def __repr__(self):
        return f"User(user_id={self.user_id}, face_embedded_vector={self.face_embedded_vector})"
