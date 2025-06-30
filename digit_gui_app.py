import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("digit_cnn_model.h5")

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")
        self.canvas = tk.Canvas(master, width=200, height=200, bg='white')
        self.canvas.pack()
        self.image = Image.new("L", (200, 200), color=255)
        self.draw = ImageDraw.Draw(self.image)
        self.label = tk.Label(master, text="", font=("Helvetica", 24))
        self.label.pack()
        self.button_frame = tk.Frame(master)
        self.button_frame.pack()
        tk.Button(self.button_frame, text="Recognise", command=self.predict_digit).pack(side="left")
        tk.Button(self.button_frame, text="Clear", command=self.clear_canvas).pack(side="right")
        self.canvas.bind("<B1-Motion>", self.paint)

    def paint(self, event):
        x, y = event.x, event.y
        self.canvas.create_oval(x-8, y-8, x+8, y+8, fill='black')
        self.draw.ellipse([x-8, y-8, x+8, y+8], fill=0)

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 200, 200], fill=255)
        self.label.config(text="")

    def predict_digit(self):
        img = self.image.resize((28, 28)).convert("L")
        img = ImageOps.invert(img)
        img = np.array(img).astype("float32") / 255.0
        img = img.reshape(1, 28, 28, 1)
        prediction = model.predict(img)
        digit = np.argmax(prediction)
        confidence = int(np.max(prediction) * 100)
        self.label.config(text=f"{digit}, {confidence}%")

root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()