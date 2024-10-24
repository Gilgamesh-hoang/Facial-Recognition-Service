import uuid

from bson import Binary
import numpy as np
from src.utils.database import get_user_collection
from src.models.user import User

def insert_user(user: User) -> str:
    """Thêm một người dùng vào MongoDB."""
    collection = get_user_collection()
    try:
        collection.insert_one(user.to_dict())
        return f"User {user.user_id} đã được thêm thành công!"
    except Exception as e:
        return f"Lỗi khi thêm người dùng: {e}"


def get_user_by_id(user_id: uuid.UUID) -> User:
    """Lấy thông tin người dùng từ MongoDB."""
    collection = get_user_collection()
    document = collection.find_one({"_id": Binary.from_uuid(user_id)})

    if document:
        return User.from_dict(document)
    else:
        raise ValueError(f"User với ID {user_id} không tồn tại.")

def get_all_users():
    collection = get_user_collection()
    documents = collection.find()
    users = []
    for document in documents:
        users.append(User.from_dict(document))
    return users


if __name__ == "__main__":
    # Tạo đối tượng User với UUID và vector nhúng
    user = User(
        user_id=uuid.uuid4(),
        face_embedded_vector=np.random.rand(128)  # Vector 128 chiều
    )

    # Thêm người dùng vào MongoDB
    # print(insert_user(user))

    # Lấy người dùng theo ID
    id = uuid.UUID("72072f76-1ecb-474f-8f26-7fa3e1120424")
    retrieved_user = get_user_by_id(id)
    print(retrieved_user)


# Ví dụ sử dụng
# if __name__ == "__main__":
#     new_user = User(
#         user_id=uuid.uuid4(),
#         face_embedded_vector=[0.12, -0.25, 0.33, 0.45, -0.67]
#     )
#
#     print(insert_user(new_user))  # Thêm user vào database
#
#     # Lấy lại user theo ID
#     try:
#         fetched_user = get_user_by_id(new_user.user_id)
#         print(f"Đã lấy được user: {fetched_user}")
#     except ValueError as e:
#         print(e)
