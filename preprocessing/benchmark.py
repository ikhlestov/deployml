import time

from skimage.transform import resize as resize_skimage
from cv2 import resize as resize_cv2
import numpy as np


np.random.seed(42)

INPUT_SIZE = (1024, 768)
OUT_WIDTH = 640
OUT_HEIGHT = 360

methods = [
    ('skimage var size', resize_skimage, OUT_WIDTH, OUT_HEIGHT),
    ('skimage same size', resize_skimage, *INPUT_SIZE),
    ('opencv var size', resize_cv2, OUT_WIDTH, OUT_HEIGHT),
    ('opencv same size', resize_cv2, *INPUT_SIZE)
]


def main():
    random_image = np.random.rand(*INPUT_SIZE, 3)
    for method_name, resize_method, width, height in methods:
        consumptions = []
        for _ in range(10):
            start_time = time.time()
            resize_method(random_image, (OUT_WIDTH, OUT_HEIGHT))
            time_cons = time.time() - start_time
            consumptions.append(time_cons)
        results = {
            'method': method_name,
            'min_cons': min(consumptions),
            'max_cons': max(consumptions),
            'mean_cons': np.mean(consumptions),
        }
        print("{method:<18} - min: {min_cons:.5f}, "
              "mean: {mean_cons:.5f}, max: {max_cons:.5f}".format(**results))


if __name__ == '__main__':
    main()
