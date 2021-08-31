from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tkinter import *

# loading Python Imaging Library
from PIL import ImageTk, Image

# To get the dialog box to open when required
from tkinter import filedialog


import argparse
import io
import time
import numpy as np

from tflite_runtime.interpreter import Interpreter

x='empty.png'

def load_labels(path):
	with open(path, 'r') as f:
		return {i: line.strip() for i, line in enumerate(f.readlines())}


def set_input_tensor(interpreter, image):
	tensor_index = interpreter.get_input_details()[0]['index']
	input_tensor = interpreter.tensor(tensor_index)()[0]
	input_tensor[:, :] = image


def classify_image(interpreter, image, top_k=1):
	"""Returns a sorted array of classification results."""
	set_input_tensor(interpreter, image)
	interpreter.invoke()
	output_details = interpreter.get_output_details()[0]
	output = np.squeeze(interpreter.get_tensor(output_details['index']))

	# If the model is quantized (uint8 data), then dequantize the results
	if output_details['dtype'] == np.uint8:
		scale, zero_point = output_details['quantization']
		output = scale * (output - zero_point)

	ordered = np.argpartition(-output, top_k)
	return [(i, output[i]) for i in ordered[:top_k]]

def open_img():
	global x
	# Select the Imagename from a folder
	x = openfilename()

	# opens the image
	img = Image.open(x)
	
	# resize the image and apply a high-quality down sampling filter
	img = img.resize((250, 250), Image.ANTIALIAS)

	# PhotoImage class is used to add image to widgets, icons etc
	img = ImageTk.PhotoImage(img)

	# create a label
	panel = Label(root, image = img)
	
	# set the image as img
	panel.image = img
	panel.grid(row=2, column = 1,columnspan = 3, rowspan = 3, padx = 5, pady = 20)

def openfilename():
	# open file dialog box to select image
	# The dialogue box has a title "Open"
	filename = filedialog.askopenfilename(title ='image')
	return filename

def capture():
	print('capture')

def classify():
	global x
	image = Image.open(x).convert('RGB').resize((width, height), Image.ANTIALIAS)

	results = classify_image(interpreter, image)
	# print(results)
	label_id, prob = results[0]
	# print(label_id)
	print(labels[label_id])
	print(prob)
	result.config(text=labels[label_id]+' '+str(round(prob*100,2))+'%')

# Create a window
root = Tk()

# Set Title as Image Loader
root.title("Image Loader")

# Set the resolution of window
root.geometry("440x400")

# Allow Window to be resizable
root.resizable(width = True, height = True)
title = Label(root, text = "Hashan Medicare").grid(row = 0, column = 0, pady = 10, columnspan = 4)
title1 = Label(root, text = "Image Classifier").grid(row = 1, column = 0, columnspan = 4)
# Create a button and place it into the window using grid layout
btn = Button(root, text ='Browse', command = open_img).grid(
										row = 2, column = 0, padx = 30)
btn1 = Button(root, text ='Capture', command = capture).grid(
										row = 3, column = 0, padx = 30)
btn1 = Button(root, text ='Classify', command = classify).grid(
										row = 4, column = 0, padx = 30)
img = Image.open('/home/pi/Desktop/bg.jpg')

# resize the image and apply a high-quality down sampling filter
img = img.resize((250, 250), Image.ANTIALIAS)

# PhotoImage class is used to add image to widgets, icons etc
img = ImageTk.PhotoImage(img)

# create a label
panel = Label(root, image = img)

# set the image as img
panel.image = img
panel.grid(row=2, column = 1,columnspan = 3, rowspan = 3, padx = 5, pady = 20)

result = Label(root, text = "Result", bg = "white", width = 43)
result.grid(row = 5, column = 0, padx = 50, columnspan = 5)

labels = load_labels('labels.txt')

interpreter = Interpreter('model.tflite')
interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']

root.mainloop()


