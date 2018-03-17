import os

BASE_SAVES_DIR = 'saves'

TENSORFLOW_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'tensorflow')
KERAS_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'keras')
PYTORCH_SAVES_DIR = os.path.join(BASE_SAVES_DIR, 'pytorch')

os.makedirs(TENSORFLOW_SAVES_DIR, exist_ok=True)
os.makedirs(KERAS_SAVES_DIR, exist_ok=True)
os.makedirs(PYTORCH_SAVES_DIR, exist_ok=True)
