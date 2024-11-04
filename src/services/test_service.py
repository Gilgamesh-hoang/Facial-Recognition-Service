from src.services.preprocessing_service import PreprocessingService


def test1(paths: list[str]):
    process = PreprocessingService()
    lst = []
    # read image from path and convert to bytes
    for path in paths:
        with open(path, 'rb') as f:
            lst.append(f.read())
    images = process.pre_process_image(lst)

    print('len', len(images))

    print(f"Type of frames: {type(images)}")  # In kiểu của `frames` (nên là list)
    if images:
        print(f"Type of frames[0]: {type(images[0])}")  # Kiểm tra phần tử đầu tiên của `frames`
        if isinstance(images[0], list):
            print(f"Type of frames[0][0]: {type(images[0][0])}")  # Nếu frames[0] là list, kiểm tra phần tử đầu tiên


if __name__ == '__main__':
    test1([
        'E:\\Facial-Recognition-Service\\Dataset\\FaceData\\processed\\rotate\\rotated1_30.png'
    ])
