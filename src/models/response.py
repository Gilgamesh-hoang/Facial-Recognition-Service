class Response:
    # define a static variable with the status code have value 2000, 2001,2002
    STATUS_CODE_SUCCESS = 5000
    STATUS_CODE_ERROR = 5001
    FACE_NOT_FOUND = 5002
    FACE_NOT_DETECTED = 5003

    def __init__(self, status_code: int, message: str, data: object = None):
        self.status_code = status_code
        self.message = message
        self.data = data

    def to_dict(self):
        return {
            "status_code": self.status_code,
            "message": self.message,
            "data": self.data
        }