from tkinter import *
import pyautogui
from PIL import Image
import cv2

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np

root = Tk()
root.title('PaintPad')
root.geometry('400x500')
root.config(bg="#CCF")

MODEL_NAME = 'MNIST-DIGITS'
IMAGE = None
Model = keras.models.load_model(MODEL_NAME)

CANVAS_WIDTH = 400
CANVAS_HEIGHT = 400

def DRAW(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords

        canvas.create_oval(x, y,
            x1, y1,
            fill='white',
            width=24,
            outline='white')

    canvas.old_coords = x, y

def RESET(event):
    canvas.old_coords = None

def PREDICT():
    x, y = canvas.winfo_rootx(), canvas.winfo_rooty()
    w, h = canvas.winfo_width(), canvas.winfo_height()
    pyautogui.screenshot('canvas.png', region=(x, y, w, h))
    canvas_im = Image.open('canvas.png').resize((28,28)).save('canvas2.png')

    canvas_image = np.array([cv2.imread('canvas2.png')[:,:,0]])
    print(np.argmax(Model.predict(canvas_image)))


canvas = Canvas(root,
    width=CANVAS_WIDTH,
    height=CANVAS_HEIGHT,
    highlightthickness=0,
    bg="black")

load = Button(
    root,
    text="Predict",
    fg="white",
    bg="darkblue",
    font="Georgia 24 bold",
    borderwidth=0,
    command=lambda: PREDICT())

clear = Button(
    root,
    text="Clear",
    fg="white",
    bg="darkred",
    font="Georgia 24 bold",
    borderwidth=0,
    command=lambda: canvas.delete('all'))

load.place(
    relx=0.3,
    rely=0.975,
    anchor='s')

clear.place(
    relx=0.75,
    rely=0.975,
    anchor='s')

canvas.pack()
canvas.old_coords = None

root.bind('<B1-Motion>', DRAW)
root.bind('<ButtonRelease-1>', RESET)
root.mainloop()
