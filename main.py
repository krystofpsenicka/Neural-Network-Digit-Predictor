# Program na předpovězení nakreslené číslice pomocí neurální sítě
# Kryštof Pšenička, I. ročník
# zimní semestr 2023/24
# NPRG030 Programování I


import tkinter as tk
import cv2
import numpy as np
from matplotlib import pyplot as plt
from data import getWeightsBiases
from predict import predict

# Get weights and biases from pretrained model
weights1, weights2, bias1, bias2 = getWeightsBiases()

class DrawingInput:
    def __init__(self):
        self.root = tk.Tk() # tkinter window
        self.root.title("Digit predictor") # change title of tkinter window
        self.canvas = tk.Canvas(self.root, width=500, height=500, bg='black') # create canvas
        self.canvas.pack(side=tk.LEFT) # add canvas to tkinter window
        self.image = np.zeros((500, 500, 3), dtype=np.uint8) # image as np array
        self.canvas.bind("<B1-Motion>", self.draw_oval) # add mouse left-click and drag event handler
        self.instructions = "Draw a digit on the black canvas and click \"Done!\" to get a prediction. You can erase the image by clicking \"Erase\"."
        self.label = tk.Label(self.root, text=self.instructions, justify=tk.LEFT, wraplength=400, width=40, font=("Arial", 15)) # create label to display instructions/prediction
        self.label.pack(pady=100) # add label to tkinter window
        self.saveBtnBorder =  tk.Frame(self.root, highlightbackground = "black",  
                         highlightthickness = 2, bd=0) # border for save button
        self.saveButton = tk.Button(self.saveBtnBorder, text="Done!", command=self.predict, font=("Arial", 15), width=40 ) # create predict button
        self.saveButton.pack()  # add save button to tkinter window
        self.saveBtnBorder.pack() # add border for save button to tkinter window
        self.eraseBtnBorder = tk.Frame(self.root, highlightbackground = "black",  
                         highlightthickness = 2, bd=0) # border for erase button
        self.eraseButton = tk.Button(self.eraseBtnBorder, text="Erase", command=self.erase, font=("Arial", 15), width=40 ) # create erase button
        self.eraseButton.pack() # add erase button to tkinter window
        self.eraseBtnBorder.pack(pady=10, padx=22) # add border for erase button to tkinter window
        self.imgData = [] # array that is saved after user clicks the save button

    def draw_oval(self, event): # function for drawing in the canvas and simultaneously into the np array image
        r = 15 # circle radius
        cv2.circle(self.image, (event.x, event.y), r, (255, 255, 255), -1) # for drawing into np array image
        self.canvas.create_oval(event.x-r, event.y-r, event.x+r, event.y+r, fill="white", outline="white") # for drawing on canvas
    
    def erase(self): #function for erasing the whole canvas
        self.image = np.zeros((500, 500, 3), dtype=np.uint8) # reinitialize the np array image
        self.canvas.delete("all") # erase everything from the tk canvas
        self.label["text"] = self.instructions # reset label
        
        

    def predict(self): # Generate prediction from self.imgData
        img = cv2.resize(self.image, (28,28)) # downscale image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # convert image to grayscale
        self.imgData = np.array(img).flatten().reshape(784, 1) # add image into self.imgData as a matrix
        if not self.imgData.any(): # image is empty
            self.label["text"]="Please draw an image on the canvas." # update label text to prompt user to draw an image
        else:  # image is not empty
            prediction = predict(self.imgData, weights1, weights2, bias1, bias2) # get prediction
            self.label["text"]=f"I think you drew a {prediction.argmax()} with {round(max(prediction)[0]*100, 2)}% certainty." # update label text to show prediction

    def run(self): # function for running the main loop (creating drawing input)
        self.root.mainloop()

if __name__ == "__main__":
    DrawingInput().run()