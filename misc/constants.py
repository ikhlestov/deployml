import os

BASE_SAVES_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), '../saves')

TENSORFLOW_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'tensorflow')
KERAS_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'keras')
PYTORCH_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'pytorch')

os.makedirs(TENSORFLOW_SAVES_DIR, exist_ok=True)
os.makedirs(KERAS_SAVES_DIR, exist_ok=True)
os.makedirs(PYTORCH_SAVES_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_SAVES_DIR, 'tensorboard'), exist_ok=True)
