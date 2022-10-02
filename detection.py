import cv2
import numpy as np
import pandas as pd
import tifffile as tif
import math

def tif2png(imagePath):
    """

    :param imagePath: path to image
    :return: void
    convert tif image to rgb image
    """
    # read tiff image as a numpy array
    img = tif.imread(imagePath)

    # check the data type
    if img.dtype == np.uint8:
        depth = 8
    elif img.dtype == np.uint16:
        depth = 16
    else:
        print("Warning: unsupported data type {}. Assuming 16-bit.", img.dtype)
        depth = 16
    # normalize the img with img.max(), make the image 8-bit, and return it
    return (img / img.max() * (2 ** 8 - 1)).astype(np.uint8)


class Image:
    # array to store all Image objects
    images = []

    def __init__(self, path):
        """

        :param path: the path where the image is stored
        """
        self.path = path  # path where the image is stored
        self.shapes = []  # a list of all shapes detected in the image
        self.image_array = []  # the numpy array representation of the image
        path_split = path.split('/')
        self.name = path_split[len(path_split)-1]


        # if the image is tiff type, convert it
        splited_path = path.split('.')
        extension = splited_path[len(splited_path)-1]
        if extension.lower() == "tif" or extension.lower() == "tiff":
            self.image_array = tif2png(path)
        else:
            self.image_array = cv2.imread(path)

    def circleDetection(self, ksize=(41, 41), dp=1, minDist=10, param1=40, param2=40, minRadius=10, maxRadius=100,
                        scale=1, sharpen=False):
        """
        detects circles in an image

        :param scale: scale (pixel/unit length)
        :param ksize: blur filter size
        :param sharpen:
        :param size: the filter kernel size used to smooth the image
        :param dp: This is the ratio of the resolution of original image to the accumulator matrix
        :param minDist: This parameter controls the minimum distance between detected circles.
        :param param1: Canny edge detection requires two parameters — minVal and maxVal.
        Param1 is the higher threshold of the two. The second one is set as Param1/2.
        :param param2: This is the accumulator threshold for the candidate detected circles. By increasing this threshold
        value, we can ensure that only the best circles, corresponding to larger accumulator values, are returned.
        :param minRadius: Minimum circle radius.
        :param maxRadius: Maximum circle radius.
        :return: none
        """

        # if the image is single channel, convert it to 3 channels
        if len(self.image_array.shape) == 2:
            self.image_array = cv2.cvtColor(self.image_array, cv2.COLOR_GRAY2BGR)
        # sharpen the img to get rid of out of focus blurring
        if sharpen:
            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]])
            self.image_array = cv2.filter2D(src=self.image_array, ddepth=-1, kernel=kernel)

        # convert to gray scale
        imgGray = cv2.cvtColor(self.image_array, cv2.COLOR_BGR2GRAY)

        # blur
        imgGrayBlurred = cv2.GaussianBlur(imgGray, ksize, 0)

        #self.image_array = cv2.GaussianBlur(imgGray, ksize, 0)

        # detect circles
        detected_circles = cv2.HoughCircles(imgGrayBlurred,
                                            cv2.HOUGH_GRADIENT, dp=dp, minDist=minDist, param1=param1,
                                            param2=param2, minRadius=minRadius, maxRadius=maxRadius)

        # Draw circles that are detected.
        if detected_circles is not None:
            circleID = 1
            # Convert the circle parameters a, b and r to integers.
            detected_circles = np.uint16(np.around(detected_circles))

            for pt in detected_circles[0, :]:
                a, b, r = pt[0], pt[1], pt[2]

                self.shapes.append(Circle(circleID, r/scale, (a, b)))

                # Draw the circumference of the circle.
                cv2.circle(self.image_array, (a, b), r, (0, 255, 0), 2)

                # Draw the circle id at the center of each circle
                cv2.putText(self.image_array, text="{}".format(circleID), org=(a, b), fontScale=1.2, color=(0, 0, 255),
                            thickness=3,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                circleID += 1

    def show(self, scale):
        """

        :param scale: the scale which the image is shown
        :return: void
        """
        if self.image_array.any() != None:
            width = int(self.image_array.shape[1] * scale)
            height = int(self.image_array.shape[0] * scale)
            cv2.imshow("Detected Circle", cv2.resize(self.image_array, (width, height)))
            cv2.moveWindow("Detected Circle", 40, 30)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


class Shape:
    def __init__(self, idnum):
        self.id = idnum
        self.area = 0


class Circle(Shape):
    def __init__(self, idnum, radius, position):
        """

        :param idnum: shape id (int)
        :param radius: radius of the circle (int)
        :param position: position of the circle (tuple)
        """
        super().__init__(idnum)
        self.radius = radius
        self.position = position
        self.name = "Circle"
        self.area = self.radius ** 2 * math.pi
