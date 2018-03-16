from skimage.transform import resize as resize_skimage
from cv2 import resize as resize_cv2
# you may use `try/except` import approach here


def resize(image, height, width, use_cv2):
    if use_cv2:
        resized = resize_cv2(image, (height, width))
    else:
        resized = resize_skimage(image, (height, width), mode='constant')
    return resized
