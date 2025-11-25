import sys
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tqdm import tqdm
import time
import os

IMG_SIZE = 160
MODEL_PATH = "cats_vs_dogs_lowram.h5"

model = keras.models.load_model(MODEL_PATH)

def load_and_prep(path):
    img = tf.keras.utils.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.utils.img_to_array(img)
    arr = arr / 255.0
    return np.expand_dims(arr, 0)

def predict(path):
    print("\n🔥 Initializing inference pipeline...\n")

    # Fake progress bar (cosmetics like TM)
    for _ in tqdm(range(100), desc="Processing image", ncols=60):
        time.sleep(0.01)

    arr = load_and_prep(path)

    pred = model.predict(arr)[0][0]  # single value
    cat_conf = (1 - pred) * 100
    dog_conf = pred * 100

    print("\n======================")
    print("📸 Image:", os.path.basename(path))
    print("======================")
    print(f"🐱 Cat Confidence : {cat_conf:.2f}%")
    print(f"🐶 Dog Confidence : {dog_conf:.2f}%")
    print("======================\n")

if __name__ == "__main__":
    predict(sys.argv[1])