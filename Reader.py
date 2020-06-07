from cv2 import cv2     #pip install opencv-python      
import numpy as np      #pip install numpy
import operator
import tensorflow as tf     #pip install tensorflow
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
from keras.utils import np_utils    #pip install keras
import matplotlib.pyplot as plt
class Reader:
    def Read(self, PICTURE_PATH):
        # PICTURE_PATH = 'D:/Code/PySem-FInal/PySem-Final/image1.jpg'
        def pre_process_image(img, skip_dilate=False):
            """Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""
            # Gaussian blur with a kernal size (height, width) of 9.
            # Note that kernal sizes must be positive and odd and the kernel must be square.
            proc = cv2.GaussianBlur(img.copy(), (9,9),0)

            # Adaptive threshold using 11 nearest neighbour pixels
            proc = cv2.adaptiveThreshold(proc,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)

            # Invert colours, so gridlines have non-zero pixel values.
            # Necessary to dilate the image, otherwise will look like erosion instead.
            proc = cv2.bitwise_not(proc, proc)
            if not skip_dilate:
                # Dilate the image to increase the size of the grid lines.
                kernel = np.array([[0., 1., 0.], [1., 1., 1.,],[0., 1., 0.]])
                kernelCopy = np.uint8(kernel)
                proc = cv2.dilate(proc, kernelCopy)
            return proc
            

        def show_image(img):
            """Shows an image until any key is pressed"""
            cv2.imshow('image', img)  # Show the image
            cv2.waitKey(0)  # Wait for any key to be pressed (with the image window active)
            cv2.destroyAllWindows()  # Close all windows

        def find_corners_of_largest_polygon(img):
            """Finds the 4 extreme corners of the largest contour in the image"""
            contours, _ = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key = cv2.contourArea, reverse=True) #sort by area descending
            polygon = contours[0] #largest image

            bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
            top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

            return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]
            
        def distance_between(p1, p2):
            """Returns the scalar distance between twho points"""
            a = p2[0] - p1[0]
            b = p2[1] - p1[1]
            return np.sqrt((a**2)+(b**2))

        def crop_and_warp(img, crop_rect):
            """"Crops and warps a rectangular section from an image into square of similar size"""
            top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
            
            src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

            #Get the longest side in the ractangle
            side = max([
                distance_between(bottom_right, top_right),
                distance_between(top_left, bottom_left),
                distance_between(bottom_right, bottom_left),
                distance_between(top_left, top_right)
            ])

            # Describe a square with side of the calculated length, this is the new perspective we want to warp to
            dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

            # Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
            m = cv2.getPerspectiveTransform(src, dst)

            # performs the transformation on the original image
            return cv2.warpPerspective(img, m, (int(side), int(side)))

        def infer_grid(img):
            """Infers 81 cell grid from a square image"""
            squares = []
            side = img.shape[:1]
            side = side[0] / 9
            for i in range(9):
                for j in range(9):
                    p1 = (i*side, j*side) #top left corner of bounding box
                    p2 = ((i+1) * side, (j+1)*side) #bottom right corner of bounding box
                    squares.append((p1,p2))
            return squares

        def display_rects(in_img, rects, color = 255):
            """Displays rectangles on the image"""
            img = in_img.copy()
            for rect in rects:
                img = cv2.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x)for x in rect[1]),color)
            show_image(img)
            return img

        def cut_from_rect(img, rect):
            """Cuts a rectangle from an image using the top left and bottom right points."""
            return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]


        def scale_and_centre(img, size, margin=0, background=0):
            """Scales and centres an image onto a new background square."""
            h, w = img.shape[:2]

            def centre_pad(length):
                """Handles centering for a given length that may be odd or even."""
                if length % 2 == 0:
                    side1 = int((size - length) / 2)
                    side2 = side1
                else:
                    side1 = int((size - length) / 2)
                    side2 = side1 + 1
                return side1, side2

            def scale(r, x):
                return int(r * x)

            if h > w:
                t_pad = int(margin / 2)
                b_pad = t_pad
                ratio = (size - margin) / h
                w, h = scale(ratio, w), scale(ratio, h)
                l_pad, r_pad = centre_pad(w)
            else:
                l_pad = int(margin / 2)
                r_pad = l_pad
                ratio = (size - margin) / w
                w, h = scale(ratio, w), scale(ratio, h)
                t_pad, b_pad = centre_pad(h)

            img = cv2.resize(img, (w, h))
            img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, None, background)
            return cv2.resize(img, (size, size))


        def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
            """
            Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
            connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
            """
            img = inp_img.copy()  # Copy the image, leaving the original untouched
            height, width = img.shape[:2]

            max_area = 0
            seed_point = (None, None)

            if scan_tl is None:
                scan_tl = [0, 0]

            if scan_br is None:
                scan_br = [width, height]

            # Loop through the image
            for x in range(scan_tl[0], scan_br[0]):
                for y in range(scan_tl[1], scan_br[1]):
                    # Only operate on light or white squares
                    if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
                        area = cv2.floodFill(img, None, (x, y), 64)
                        if area[0] > max_area:  # Gets the maximum bound area which should be the grid
                            max_area = area[0]
                            seed_point = (x, y)

            # Colour everything grey (compensates for features outside of our middle scanning range
            for x in range(width):
                for y in range(height):
                    if img.item(y, x) == 255 and x < width and y < height:
                        cv2.floodFill(img, None, (x, y), 64)

            mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

            # Highlight the main feature
            if all([p is not None for p in seed_point]):
                cv2.floodFill(img, mask, seed_point, 255)

            top, bottom, left, right = height, 0, width, 0

            for x in range(width):
                for y in range(height):
                    if img.item(y, x) == 64:  # Hide anything that isn't the main feature
                        cv2.floodFill(img, mask, (x, y), 0)

                    # Find the bounding parameters
                    if img.item(y, x) == 255:
                        top = y if y < top else top
                        bottom = y if y > bottom else bottom
                        left = x if x < left else left
                        right = x if x > right else right

            bbox = [[left, top], [right, bottom]]
            return img, np.array(bbox, dtype='float32'), seed_point


        def extract_digit(img, rect, size):
            """Extracts a digit (if one exists) from a Sudoku square."""

            digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

            # Use fill feature finding to get the largest feature in middle of the box
            # Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
            h, w = digit.shape[:2]
            margin = int(np.mean([h, w]) / 2.5)
            _, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
            digit = cut_from_rect(digit, bbox)

            # Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
            w = bbox[1][0] - bbox[0][0]
            h = bbox[1][1] - bbox[0][1]

            # Ignore any small bounding boxes
            if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
                return scale_and_centre(digit, size, 4)
            else:
                return np.zeros((size, size), np.uint8)

        def get_digits(img, squares, size):
            """Extracts digits from their cells and builds an array"""
            digits = []
            img = pre_process_image(img.copy(), skip_dilate=True)
            for square in squares:
                digits.append(extract_digit(img, square, size))
            return digits
            
        def show_digits(digits, colour=255):
            """Shows list of 81 extracted digits in a grid format"""
            rows = []
            with_border = [cv2.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv2.BORDER_CONSTANT, None, colour) for img in digits]
            for i in range(9):
                row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=0)
                rows.append(row)
            return (np.concatenate(rows, axis=1))

        #Reads the picture as a numpy array
        img = cv2.imread(PICTURE_PATH, cv2.IMREAD_GRAYSCALE)
        show_image(img)

        #First step, we make the image blurry and binary, to easyly find corners
        processed = pre_process_image(img)
        show_image(processed)

        #The we find corners
        corners = find_corners_of_largest_polygon(processed)

        #Crop and Warp the image, so that the main grid is the only thing visible
        cropped = crop_and_warp(img, corners)
        show_image(cropped)

        # Marks a gridline to see where are numbers located
        squares = infer_grid(cropped)
        display_rects(cropped, squares)

        # Extracts a cell if there is a number inside
        digits = get_digits(cropped, squares, 28)
        dig = show_digits(digits)
        show_image(dig)

        import tensorflow as tf
        import pickle
        import os
        tf.compat.v1.disable_eager_execution()
        x = tf.compat.v1.placeholder(tf.float32, shape=[0, 784])  # Placeholder for input
        y = tf.compat.v1.placeholder(tf.float32, shape=[0, 10])  # Placeholder for true labels (used in training)
        hidden_neurons = 16  # Number of neurons in the hidden layer, constant

        def weights(shape):
            """Weight initialisation with a random, slightly positive value to help prevent dead neurons."""
            return tf.Variable(tf.random.normal(shape, stddev=0.1))
        def biases(shape):
            """Bias initialisation with a positive constant, helps to prevent dead neurons."""
            return tf.Variable(tf.constant(0.1, shape=shape))

        # Hidden layer
        w_1 = weights([784, hidden_neurons])
        b_1 = biases([hidden_neurons])
        h_1 = tf.nn.sigmoid(tf.matmul(x, w_1) + b_1)  # Order of x and w_1 matters here purely syntactically

        # Output layer
        w_2 = weights([hidden_neurons, 10])
        b_2 = biases([10])
        y = tf.matmul(h_1, w_2) + b_2  # Note that we don't use sigmoid here because the next step uses softmax

        # Cross entropy cost function
        # cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

        # Gradient descent and backpropagation learning
        # train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cost)

        # Accuracy comparison/measurement function
        # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        #accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        def load_data(file_name):
            """Loads Python object from disk."""
            with open(file_name, 'rb') as f:
                data = pickle.load(f)
            return data

        # Train the network
        # model_path = 'D:/Code/PySem-FInal/PySem-Final/'
        ds = load_data(os.path.join('D:/Code/PySem-FInal/PySem-Final/neural_net', 'digit-basic'))  # Dataset

        model = keras.Sequential()
        model.add(Flatten(input_shape=(28, 28)))
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dense(10, activation=tf.nn.softmax))


        model.compile(loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy'])
        ds.train.images = np.reshape(ds.train.images, (-1,28,28))
        model.fit(ds.train.images, ds.train.labels, epochs=10)

        resize = np.reshape(digits,(-1,28,28))                  # Calculate prediction for test data
                     
        predictions = model.predict(resize)


        puzzleArray = [[],[],[],[],[],[],[],[],[]]

            #Places all the values into a list of lists
        ini = 0
        for x in range(9):
            for y in range(9):
                if max(predictions[ini]) < 0.6:
                    puzzleArray[y].append(0)
                    ini+=1
                else:
                    puzzleArray[y].append(np.argmax(predictions[ini]))
                    ini+=1

            #The puzzleArray converted to string, that will be passed further for solving
        puzzleString = ""
        for col in range(9):
            if col == 3 or col == 6:
                puzzleString += "------+------+------\n"
            
            for row in range(9):
                if row == 3 or row ==6:
                    puzzleString += "|"
                    
                puzzleString += (str)(puzzleArray[col][row])
                puzzleString += " "
            puzzleString += '\n'
        return puzzleString






