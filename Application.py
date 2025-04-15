import numpy as np
import cv2
import os
import operator
import time
from string import ascii_uppercase
import tkinter as tk
from PIL import Image, ImageTk
from spellchecker import SpellChecker
from keras.models import model_from_json
import tensorflow as tf

# Set TensorFlow to use GPU if available
os.environ["THEANO_FLAGS"] = "device=cuda, assert_no_cpu_op=True"
print(tf.__version__)

# Application Class
class Application:
    def __init__(self):
        # Initialize variables
        self.spell = SpellChecker()
        self.vs = cv2.VideoCapture(0)
        self.current_image = None
        self.current_image2 = None

        # Load models
        self.load_models()

        # Setup UI
        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("900x900")

        self.setup_ui()

        # Initialize variables
        self.str = ""  # Full sentence
        self.word = ""  # Current word
        self.current_symbol = "Empty"  # Last recognized symbol
        self.blank_flag = 0  # Flag to handle blank space
        self.ct = {char: 0 for char in ascii_uppercase}  # Character count
        self.ct['blank'] = 0

        self.video_loop()

    def load_models(self):
        # Loading all models
        self.model_paths = {
            "model_new": "Models/model_new.json",
            "model-bw_dru": "Models/model-bw_dru.json",
            "model-bw_tkdi": "Models/model-bw_tkdi.json",
            "model-bw_smn": "Models/model-bw_smn.json"
        }
        self.loaded_models = {}
        for model_name, model_path in self.model_paths.items():
            with open(model_path, "r") as json_file:
                model_json = json_file.read()
                model = model_from_json(model_json)
                weights_path = model_path.replace("json", "weights.h5")
                model.load_weights(weights_path)
                self.loaded_models[model_name] = model

        print("Models loaded successfully")

    def setup_ui(self):
        # UI Elements
        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root, text="Sign Language To Text Conversion", font=("Courier", 30, "bold"))
        self.T.place(x=60, y=5)

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root, text="Character :", font=("Courier", 30, "bold"))
        self.T1.place(x=10, y=540)

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root, text="Word :", font=("Courier", 30, "bold"))
        self.T2.place(x=10, y=595)

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root, text="Sentence :", font=("Courier", 30, "bold"))
        self.T3.place(x=10, y=645)

        self.T4 = tk.Label(self.root, text="Suggestions :", fg="red", font=("Courier", 30, "bold"))
        self.T4.place(x=250, y=690)

        self.bt1 = tk.Button(self.root, command=self.action1, height=0, width=0)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=0, width=0)
        self.bt2.place(x=325, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=0, width=0)
        self.bt3.place(x=625, y=745)

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.flip(frame, 1)

            # Define region of interest (ROI)
            x1 = int(0.5 * frame.shape[1])
            y1 = 10
            x2 = frame.shape[1] - 10
            y2 = int(0.5 * frame.shape[1])

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 1)
            cv2image = cv2.cvtColor(cv2image, cv2.COLOR_BGR2RGBA)

            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            cv2image = cv2image[y1:y2, x1:x2]

            # Preprocess image for prediction
            gray = cv2.cvtColor(cv2image, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)
            th3 = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            ret, res = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            self.predict(res)

            self.current_image2 = Image.fromarray(res)
            imgtk = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk
            self.panel2.config(image=imgtk)

            # Update UI with current symbol, word, and sentence
            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))

            # Generate word suggestions
            predicts = sorted(self.spell.candidates(self.word))
            self.update_suggestions(predicts)

        self.root.after(5, self.video_loop)

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        result = self.loaded_models["model_new"].predict(test_image.reshape(1, 128, 128, 1))
        prediction = self.get_prediction(result)
        self.current_symbol = prediction[0][0]

        # Update the character count and word formation logic
        if self.current_symbol == "blank":
            self.ct["blank"] = 0
            for char in ascii_uppercase:
                self.ct[char] = 0
        else:
            self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:
            self.ct["blank"] = 0
            for char in ascii_uppercase:
                self.ct[char] = 0

            if self.current_symbol == "blank":
                if self.blank_flag == 0:
                    self.blank_flag = 1
                    if len(self.str) > 0:
                        self.str += " "
                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""
                self.blank_flag = 0
                self.word += self.current_symbol

    def get_prediction(self, result):
        prediction = {ascii_uppercase[i]: result[0][i+1] for i in range(26)}
        prediction = sorted(prediction.items(), key=operator.itemgetter(1), reverse=True)
        return prediction

    def update_suggestions(self, predicts):
        if len(predicts) > 1:
            self.bt1.config(text=list(predicts)[0], font=("Courier", 20))
        else:
            self.bt1.config(text="")

        if len(predicts) > 2:
            self.bt2.config(text=list(predicts)[1], font=("Courier", 20))
        else:
            self.bt2.config(text="")

        if len(predicts) > 3:
            self.bt3.config(text=list(predicts)[2], font=("Courier", 20))
        else:
            self.bt3.config(text="")

    def action1(self):
        self.apply_suggestion(0)

    def action2(self):
        self.apply_suggestion(1)

    def action3(self):
        self.apply_suggestion(2)

    def apply_suggestion(self, index):
        predicts = list(self.spell.candidates(self.word))
        if index < len(predicts):
            self.word = ""
            self.str += " "
            self.str += predicts[index]

    def destructor(self):
        print("Closing application...")
        self.vs.release()
        cv2.destroyAllWindows()
        self.root.destroy()

# Run the application
print("Starting application...")
Application().root.mainloop()
