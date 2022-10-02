from detection import *
import os
import cv2
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from PIL import Image as Ig
import pandas as pd
import numpy as np
import tifffile as tif
import math

# class for the main frame
class MainFrame(ttk.Frame):
    # a list of selected images
    images = []
    image_id = 0

    def __init__(self, parent, controller):
        super().__init__(parent)
        options = {'padx': 5, 'pady': 5}

        # save frame controller as a attribute
        self.controller = controller

        # parent frame/window
        self.parent = parent

        # select images button
        self.select_image_button = ttk.Button(self, text="Select Images", command=self.selectImage)
        self.select_image_button.grid(column=0, row=0, **options)

        self.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)

    def selectImage(self):
        FrameController.frame_num += 1
        self.controller.showFrame()
        filetypes = (('TIF files', '*.TIF'), ('All files', '*.*'))
        filenames = fd.askopenfilenames(title='open files', initialdir='/', filetypes=filetypes)
        showinfo(title="Selected Files", message=filenames)
        MainFrame.images = filenames


# frame for images
class ImageFrame(ttk.Frame):
    # store paths of images
    image_paths = []
    # store Image objects
    images = []
    image_id = 0

    def __init__(self, parent, controller):
        super().__init__(parent)
        options = {'padx': 5, 'pady': 5}
        self.disabled_buttons = []
        self.parent = parent

        self.controller = controller

        # start detection button
        self.start_detection_button = ttk.Button(self, text="Start Detection",
                                                 command=self.startDetection)
        self.start_detection_button.grid(column=0, row=0, **options, sticky=tk.W)

        # next button
        self.next_button = ttk.Button(self, text="Next Image", command=self.nextImage, state="disabled")
        self.next_button.grid(column=0, row=1, **options, sticky=tk.W)
        self.disabled_buttons.append(self.next_button)

        # previous button
        self.previous_button = ttk.Button(self, text="Previous Image", command=self.previousImage, state="disabled")
        self.previous_button.grid(**options, sticky=tk.W)
        self.disabled_buttons.append(self.previous_button)

        # detect again button
        self.detect_again_button = ttk.Button(self, text="Detect again", command=self.detectAgain, state="disabled")
        self.detect_again_button.grid(**options, sticky=tk.W)
        self.disabled_buttons.append(self.detect_again_button)

        # save button
        self.save_button = ttk.Button(self, text="Save", command=self.save, state="disabled")
        self.save_button.grid(**options, sticky=tk.W, column=0, row=6)
        self.disabled_buttons.append(self.save_button)

        # finish button, go back to the main frame
        self.finish_button = ttk.Button(self, text="Finish", command=self.finish)
        self.finish_button.grid(**options, sticky=tk.W, row=7, column=0)

        #### parameters ####
        self.param1 = tk.StringVar()  # stringVar has "" as a default string
        self.param1_entry = ttk.Entry(self, textvariable=self.param1)
        self.param1_entry.grid(column=2, row=0, sticky=tk.W, **options)
        self.param1_label = ttk.Label(self, text="parameter 1")
        self.param1_label.grid(column=1, row=0, sticky=tk.W)

        self.param2 = tk.StringVar()
        self.param2_entry = ttk.Entry(self, textvariable=self.param2)
        self.param2_entry.grid(column=2, row=1, sticky=tk.W, **options)
        self.param2_label = ttk.Label(self, text="parameter 2")
        self.param2_label.grid(column=1, row=1, sticky=tk.W)

        self.max_radius = tk.StringVar()
        self.max_radius_entry = ttk.Entry(self, textvariable=self.max_radius)
        self.max_radius_entry.grid(column=2, row=2, sticky=tk.W, **options)
        self.max_radius_label = ttk.Label(self, text="max radius")
        self.max_radius_label.grid(column=1, row=2, sticky=tk.W)

        self.min_radius = tk.StringVar()
        self.min_radius_entry = ttk.Entry(self, textvariable=self.min_radius)
        self.min_radius_entry.grid(column=2, row=3, sticky=tk.W, **options)
        self.min_radius_label = ttk.Label(self, text="min radius")
        self.min_radius_label.grid(column=1, row=3, sticky=tk.W)

        self.blur = tk.StringVar()
        self.blur_entry = ttk.Entry(self, textvariable=self.blur)
        self.blur_entry.grid(column=2, row=4, sticky=tk.W, **options)
        self.blur_label = ttk.Label(self, text="Blur size [odd integer]")
        self.blur_label.grid(column=1, row=4, sticky=tk.W)

        # user input for scale
        self.scale = tk.StringVar()
        if self.scale.get() == "":
            self.scale.set("1")
        self.scale_entry = ttk.Entry(self, textvariable=self.scale)
        self.scale_entry.grid(column=2, row=6, sticky=tk.W)
        self.scale_label = ttk.Label(self, text="Scale (pixel/unit length)")
        self.scale_label.grid(column=1, row=6, sticky=tk.W)

        # user input for unit
        self.unit = tk.StringVar()
        if self.unit.get() == "":
            self.unit.set("pixel")
        self.unit_entry = ttk.Entry(self, textvariable=self.unit)
        self.unit_entry.grid(column=2, row=7, sticky=tk.W)
        self.unit_label = ttk.Label(self, text="unit (ex: um, nm)")
        self.unit_label.grid(column=1, row=7, sticky=tk.W)

        self.grid(row=0, column=0, padx=10, pady=10, sticky=tk.NSEW)

    def nextImage(self):
        # if the next image exists in the already detected image array
        if ImageFrame.image_id < len(ImageFrame.images) - 1:
            ImageFrame.image_id += 1
            img_show = Ig.fromarray(ImageFrame.images[ImageFrame.image_id].image_array)
            self.showImage(img_show)
        # if the next image exists in the image paths array, do the detection
        # on the next image
        elif ImageFrame.image_id < len(ImageFrame.image_paths) - 1:
            ImageFrame.image_id += 1
            img = Image(ImageFrame.image_paths[ImageFrame.image_id])
            if (self.param1_entry.get() == "") or (self.param2_entry.get() == "") or (
                    self.max_radius_entry.get() == "") or (self.min_radius_entry.get() == "") or (
                    self.blur.get() == ""
            ):
                img.circleDetection(scale=int(self.scale_entry.get()))
            else:
                img.circleDetection(param1=int(self.param1_entry.get()), param2=int(self.param2_entry.get()),
                                    minRadius=int(self.min_radius_entry.get()),
                                    maxRadius=int(self.max_radius_entry.get()),
                                    ksize=(int(self.blur.get()), int(self.blur.get())),
                                    scale=int(self.scale_entry.get()))

            ImageFrame.images.append(img)
            # turn the numpy array into PIL Image
            img_show = Ig.fromarray(ImageFrame.images[ImageFrame.image_id].image_array)
            self.showImage(img_show)

    def previousImage(self):
        # if not at the first picture already
        if ImageFrame.image_id != 0:
            ImageFrame.image_id -= 1
            img_show = Ig.fromarray(ImageFrame.images[ImageFrame.image_id].image_array)
            self.showImage(img_show)

    def finish(self):
        FrameController.frame_num = 0
        self.start_detection_button['state'] = "!disabled"
        self.reset()
        self.controller.showFrame()

    def startDetection(self):
        try:
            if len(MainFrame.images) != 0:
                # save the paths of images in this class
                ImageFrame.image_paths = MainFrame.images

                # detect the image
                img = Image(ImageFrame.image_paths[ImageFrame.image_id])
                if (self.param1_entry.get() == "") or (self.param2_entry.get() == "") or (
                        self.max_radius_entry.get() == "") or (self.min_radius_entry.get() == "") or (
                        self.blur.get() == ""
                ):
                    img.circleDetection(scale=int(self.scale_entry.get()))
                else:
                    img.circleDetection(param1=int(self.param1_entry.get()), param2=int(self.param2_entry.get()),
                                        minRadius=int(self.min_radius_entry.get()),
                                        maxRadius=int(self.max_radius_entry.get()),
                                        ksize=(int(self.blur.get()), int(self.blur.get())),
                                        scale=int(self.scale_entry.get()))

                # add the detected img to the detected image array
                ImageFrame.images.append(img)
                # turn the numpy array into PIL Image
                img_show = Ig.fromarray(ImageFrame.images[ImageFrame.image_id].image_array)
                self.showImage(img_show)

                # enable some buttons
                for button in self.disabled_buttons:
                    button["state"] = "!disabled"

                self.start_detection_button['state'] = "disabled"

        except:
            print("Error")

    def showImage(self, PIL_image):
        PIL_image.show()

    # detect current image again
    def detectAgain(self):
        img = ImageFrame.image_paths[ImageFrame.image_id]
        img = Image(img)
        if (self.param1_entry.get() == "") or (self.param2_entry.get() == "") or (
                self.max_radius_entry.get() == "") or (self.min_radius_entry.get() == "") or (
                self.blur.get() == ""
        ):
            img.circleDetection(scale=int(self.scale_entry.get()))
        else:
            img.circleDetection(param1=int(self.param1_entry.get()), param2=int(self.param2_entry.get()),
                                minRadius=int(self.min_radius_entry.get()),
                                maxRadius=int(self.max_radius_entry.get()),
                                ksize=(int(self.blur.get()), int(self.blur.get())),
                                scale=int(self.scale_entry.get()))

        img_show = Ig.fromarray(img.image_array)
        ImageFrame.images[ImageFrame.image_id] = img
        self.showImage(img_show)

    # reset everything
    def reset(self):
        ImageFrame.image_id = 0
        ImageFrame.images = []
        ImageFrame.image_paths = 0

        for button in self.disabled_buttons:
            button["state"] = "disabled"

    def save(self):
        # if there are analyzed images in the images array
        if len(ImageFrame.images) != 0:
            # create a pandas dataframe
            data = pd.DataFrame(columns=["File name",
                                         "Shape types",
                                         "Radius ({})".format(self.unit.get()),
                                         "Area ({}^2)".format(self.unit.get()),
                                         "Position (x, y)"])

            counter = 0

            # go through each image to and add info to the dataframe
            for image in ImageFrame.images:
                for shape in image.shapes:
                    data.loc[counter] = [image.name] + [shape.name] + [shape.radius, shape.area, shape.position]
                    counter += 1



            print(data)
            data.to_csv("results.csv", index=False)


class FrameController(ttk.LabelFrame):
    # frame_num tracks which frame should be shown
    frame_num = 0

    def __init__(self, parent):
        super().__init__(parent)

        # initialize frames
        self.frames = {0: MainFrame(parent, self), 1: ImageFrame(parent, self)}
        self.showFrame()

    # function to handel frame switch
    def showFrame(self):
        frame = self.frames[self.frame_num]
        frame.tkraise()


# class for the root window
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Image Analysis program')
        # self.geometry("500x500")


if __name__ == '__main__':
    app = App()
    FrameController(app)
    app.mainloop()
