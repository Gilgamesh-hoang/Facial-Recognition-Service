from src.services.model_service import ModelService
from src.services.preprocessing_service import PreprocessingService


def test1(paths: list[str]):
    process = PreprocessingService()
    lst = []
    # read image from path and convert to bytes
    for path in paths:
        with open(path, 'rb') as f:
            lst.append(f.read())
    images = process.pre_process_image(lst)

    #     print len of item in images
    for image in images:
        print('len', len(image))


if __name__ == '__main__':
    test1([
        'C:\\Users\\FPT SHOP\\Pictures\\Saved Pictures\\IMG_20220820_063751.jpg',
    ])
