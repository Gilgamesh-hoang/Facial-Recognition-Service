from pymongo import MongoClient

def get_mongo_client() -> MongoClient:
    """Kết nối đến MongoDB và trả về client."""
    try:
        client = MongoClient('mongodb://root:mongodb@localhost:27017/', uuidRepresentation='standard')
        print("Kết nối đến MongoDB thành công!")
        return client
    except Exception as e:
        print(f"Lỗi khi kết nối đến MongoDB: {e}")
        raise

def get_user_collection():
    """Trả về collection 'user' trong database 'snapgram'."""
    client = get_mongo_client()
    db = client['snapgram']
    return db['user']
