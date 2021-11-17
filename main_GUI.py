from rotate import *

import tkinter as tk
from tkinter import font
import RPi.GPIO as GPIO
import picamera
import threading
from io import BytesIO
from PIL import Image
import time
import numpy as np
import tensorflow as tf


class MainScreen:
    def __init__(self):
        self.window = tk.Tk()  # Tkinter로 화면 생성.
        self.window.attributes('-fullscreen', True)  # 전체 화면으로 설정.
        self.fullScreenState = False  # 전체 화면 상태를 컨트롤하기 위한 변수.
        self.window.bind("<F11>", self.toggleFullScreen)  # F11을 누르면 전체 화면 토글.
        self.window.bind("<Escape>", self.quitFullScreen)  # ESC를 누르면 전체 화면 종료.
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)  # Execute on_closing fuction before exit.

        self.myFont = font.Font(family='Helvetica', size=18, weight='bold')
        self.thread_flag = False
        self.manual_state = -1

        # Set Button
        self.canButton = tk.Button(self.window, text="Can", font=self.myFont, command=self.canClick)
        self.glassButton = tk.Button(self.window, text="Glass", font=self.myFont, command=self.glassClick)
        self.paperButton = tk.Button(self.window, text="Paper", font=self.myFont, command=self.paperClick)
        self.petButton = tk.Button(self.window, text="Pet", font=self.myFont, command=self.petClick)
        self.plasticButton = tk.Button(self.window, text="Plastic", font=self.myFont, command=self.plasticClick)
        self.exitButton = tk.Button(self.window, text="x", font=self.myFont, command=self.on_closing)

        self.canButton.place(x=440, y=120, width=140, height=80)
        self.glassButton.place(x=600, y=120, width=140, height=80)
        self.paperButton.place(x=440, y=220, width=140, height=80)
        self.petButton.place(x=600, y=220, width=140, height=80)
        self.plasticButton.place(x=440, y=320, width=140, height=80)
        self.exitButton.place(x=760, y=10, width=30, height=30)

        # Set Label
        self.label_text = tk.StringVar()
        self.lb = tk.Label(textvariable=self.label_text, font=self.myFont)
        self.lb.place(x=160, y=100)

        # Set Radiobutton
        self.r = tk.IntVar()
        self.autoRadio = tk.Radiobutton(
            self.window,
            text="Automatic",
            font=self.myFont,
            variable=self.r,
            value=1,
            command=self.selectRadio
        )
        self.manuRadio = tk.Radiobutton(
            self.window,
            text="Manual",
            font=self.myFont,
            variable=self.r,
            value=2,
            command=self.selectRadio
        )
        self.autoRadio.place(x=20, y=20, width=200, height=80)
        self.manuRadio.place(x=200, y=20, width=160, height=80)
        self.autoRadio.invoke()

        # Set motor pins.
        self.motor_pins1 = [12, 16, 20, 21]  # IN1(OUT1), IN2(OUT2), IN3(OUT3), IN4(OUT4)
        self.motor_pins2 = [6, 13, 19, 26]  # IN1(OUT1), IN2(OUT2), IN3(OUT3), IN4(OUT4)

        # Set GPIO.
        print("Start GPIO setting.")
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        GPIO.cleanup()
        for i in range(4):
            GPIO.setup(self.motor_pins1[i], GPIO.OUT, initial=GPIO.LOW)
            GPIO.setup(self.motor_pins2[i], GPIO.OUT, initial=GPIO.LOW)
        print("GPIO setting is completed.")

        # Set PiCamera.
        print("Start Pi Camera setting.")
        self.image_size = 224
        self.camera_size = self.image_size * 3
        self.camera = picamera.PiCamera(resolution=(self.camera_size, self.camera_size), framerate=90)
        self.camera.start_preview(fullscreen=False, window=(60, 140, 300, 300))
        print("Pi Camera setting is completed.")

        # Set class names and load CNN model.
        print("Start CNN model loading.")
        self.class_names = ['can', 'glass', 'nothing', 'paper', 'pet', 'plastic']
        self.model_h5 = 'models/MobileNetV3(large).h5'  # Set name of CNN model file.
        self.model = None
        try:
            self.model = tf.keras.models.load_model(self.model_h5)  # Load CNN model.
            print("CNN model loading is completed.")
        except Exception as e:
            print(e)
            print("CNN model loading failed.")

        # Start predict.
        print("Start automatic garbage sorting.")
        self.current_angle = 2

        # 이 부분을 스레드로 실행해야 while문이 따로 동작되어 GUI 사용 가능.
        self.t1 = threading.Thread(target=self.predict_thread, args=())
        self.t1.daemon = True
        self.t1.start()
        
        self.t2 = threading.Thread(target=self.predict_thread, args=())
        self.t2.daemon = True
        self.t2.start()

        self.window.mainloop()  # Exit GUI

    def on_closing(self):  # Thread value check before exit.
        self.model = None

        # Pi Camera and GPIO clear.
        self.camera.stop_preview()
        self.camera.close()
        GPIO.cleanup()

        self.window.destroy()

    def toggleFullScreen(self, event):  # <F11>버튼, 전체화면 토글
        self.fullScreenState = not self.fullScreenState
        self.window.attributes("-fullscreen", self.fullScreenState)

    def quitFullScreen(self, event):  # <ESC>버튼, 전체화면 종료
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)

    def predict_thread(self):  # 인공지능 쓰레드 함수
        while self.model:
            if self.thread_flag:
                # Pi Camera set up.
                stream = BytesIO()
                self.camera.capture(stream, format='jpeg')
                stream.seek(0)
                img = Image.open(stream)
                img = img.crop((self.image_size, self.image_size, self.image_size * 2, self.image_size * 2))

                # Predict image.
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array = tf.expand_dims(img_array, 0)
                predictions = self.model.predict(img_array)
                score = tf.nn.softmax(predictions[0])
                prediction = np.argmax(score)
                self.label_text.set(f"{self.class_names[prediction]} {100 * np.max(score):.2f}%")

                if prediction == 2:  # Skip for nothing class.
                    continue

                # Save predicted image.
                img.save(f"garbage_images/{int(time.time())}_{self.class_names[prediction]}.jpg")

                # Operate the motor.
                if prediction > 2:  # Except for the nothing class, the order number is determined.
                    prediction -= 1
                self.current_angle = move_motor(self.current_angle, prediction, self.motor_pins1, self.motor_pins2)
            elif self.manual_state > -1:
                self.current_angle = move_motor(self.current_angle, self.manual_state, self.motor_pins1, self.motor_pins2)
                self.manual_state = -1

    def selectRadio(self):  # 라디오 버튼 이벤트
        if self.r.get() == 1:
            self.label_text.set("Auto")
            self.thread_flag = True
            self.canButton['state'] = tk.DISABLED
            self.glassButton['state'] = tk.DISABLED
            self.paperButton['state'] = tk.DISABLED
            self.petButton['state'] = tk.DISABLED
            self.plasticButton['state'] = tk.DISABLED
        elif self.r.get() == 2:
            self.label_text.set("Manual")
            self.thread_flag = False
            self.canButton['state'] = tk.NORMAL
            self.glassButton['state'] = tk.NORMAL
            self.paperButton['state'] = tk.NORMAL
            self.petButton['state'] = tk.NORMAL
            self.plasticButton['state'] = tk.NORMAL

    def canClick(self):
        self.label_text.set('Can')
        self.manual_state = 0

    def glassClick(self):
        self.label_text.set('Glass')
        self.manual_state = 1

    def paperClick(self):
        self.label_text.set('Paper')
        self.manual_state = 2

    def petClick(self):
        self.label_text.set('Pet')
        self.manual_state = 3

    def plasticClick(self):
        self.label_text.set('Plastic')
        self.manual_state = 4


if __name__ == "__main__":
    try:
        app = MainScreen()
    except Exception as e:
        print(e)
    print("End all jobs.")
