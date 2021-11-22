from rotate import *

import RPi.GPIO as GPIO
import picamera
from io import BytesIO
from PIL import Image

import numpy as np
import tensorflow as tf

# motor pin set up
motor_pins1 = [12, 16, 20, 21]  # IN1(OUT1), IN2(OUT2), IN3(OUT3), IN4(OUT4)
motor_pins2 = [6, 13, 19, 26]  # IN1(OUT1), IN2(OUT2), IN3(OUT3), IN4(OUT4)

# Set motor
print("Start GPIO setting.")
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.cleanup()
for i in range(4):
    GPIO.setup(motor_pins1[i], GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(motor_pins2[i], GPIO.OUT, initial=GPIO.LOW)
print("GPIO setting is completed.")

# Set camera
print("Start Pi Camera setting.")
image_size = 224
capture_size = image_size * 3
camera = picamera.PiCamera(resolution=(capture_size, capture_size), framerate=90)
camera.start_preview(fullscreen=False, window=(10, 40, 300, 300))
print("Pi Camera setting is completed.")

# Set class names and load cnn model
print("Start CNN model loading.")
class_names = ['cans', 'colorless_pet', 'glass', 'nothing', 'paper', 'plastic']
model_h5 = 'models/MobileNetV3(large).h5'  # 불러올 모델 파일의 이름을 정한다.
model = None
try:
    model = tf.keras.models.load_model(model_h5)  # 모델을 불러온다.
    print("CNN model loading is completed.")
except Exception as e:
    print(e)
    print("CNN model loading failed.")

# Automatic garbage sorting.
print("Start automatic garbage sorting.")
prediction_count = 0
current_angle = 2
while model:
    # Pi Camera set up.
    stream = BytesIO()
    camera.capture(stream, format='jpeg')
    stream.seek(0)
    img = Image.open(stream)
    img = img.crop((image_size, image_size, image_size * 2, image_size * 2))

    # Predict image.
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])
    prediction = np.argmax(score)
    print(f"{class_names[prediction]} {100 * np.max(score):.2f}%")

    if prediction == 2:  # Skip for nothing class.
        continue

    # Save predicted image.
    img.save(f"garbage_images/{int(time.time())}_{class_names[prediction]}.jpg")

    # Operate the motor.
    if prediction > 2:  # Except for the nothing class, the order number is determined.
        prediction -= 1
    current_angle = move_motor(current_angle, prediction, motor_pins1, motor_pins2)

# Pi Camera and GPIO clear
camera.stop_preview()
camera.close()
GPIO.cleanup()
print("End all jobs.")
