import RPi.GPIO as GPIO
import picamera
from tensorflow.keras import models
from io import BytesIO
from PIL import Image
import numpy as np
from rotate import *

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
class_names = ['can', 'glass', 'nothing', 'paper', 'pet', 'plastic']
model_h5 = 'models/MobileNetV2.h5'  # 불러올 모델 파일의 이름을 정한다.
model = None
try:
    model = models.load_model(model_h5)  # 모델을 불러온다.
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
    image = Image.open(stream)
    image = image.crop((image_size, image_size, image_size * 2, image_size * 2))

    # Predict image.
    image_array = np.array(image)
    image_array = (np.expand_dims(image_array, 0))
    predictions = model.predict(image_array)
    prediction = np.argmax(predictions[0])
    print(f"{100 * np.max(predictions[0]):2.0f}% {class_names[prediction]}")

    if prediction == 2:  # Skip for nothing class.
        continue

    # Save predicted image.
    image.save(f"garbage_images/{int(time.time())}_{class_names[prediction]}.jpg")

    # Operate the motor.
    if prediction > 2:  # Except for the nothing class, the order number is determined.
        prediction -= 1
    current_angle = move_motor(current_angle, prediction, motor_pins1, motor_pins2)

# Pi Camera and GPIO clear
camera.stop_preview()
camera.close()
GPIO.cleanup()
print("End all jobs.")
