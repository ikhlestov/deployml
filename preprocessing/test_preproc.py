import pytest
import numpy as np

from image_preproc import resize


@pytest.fixture
def input_size():
    return (1024, 768)


@pytest.fixture
def out_width():
    return 640


@pytest.fixture
def out_height():
    return 360


@pytest.fixture
def random_image(input_size):
    return np.random.rand(*input_size, 3)


def test_skimage_resize(random_image, out_width, out_height):
    result = resize(random_image, height=out_height, width=out_width, use_cv2=False)
    res_height, res_width, _ = result.shape
    assert res_height == out_height
    assert res_width == out_width


def test_cv2_resize(random_image, out_width, out_height):
    result = resize(random_image, height=out_height, width=out_width, use_cv2=True)
    res_height, res_width, _ = result.shape
    assert res_height == out_height
    assert res_width == out_width


@pytest.mark.parametrize('use_cv2', [False, True])
def test_resizers(random_image, out_width, out_height, use_cv2):
    result = resize(random_image, height=out_height, width=out_width, use_cv2=use_cv2)
    res_height, res_width, _ = result.shape
    assert res_height == out_height
    assert res_width == out_width
