import uuid


def is_valid_uuid(uuid_string: str):
    try:
        val = uuid.UUID(uuid_string)
        return True
    except ValueError:
        return False
