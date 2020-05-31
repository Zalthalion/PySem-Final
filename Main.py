# pip install opencv-python
from cv2 import cv2
# pip install keras
# pip install sckit-learn
# pip imstall numpy
import numpy as np
#pip install matplotlib
from matplotlib import pyplot as plt
import operator

def plot_many_images(images, titles, rows=1, columns=2):
	"""Plots each image in a given list in a grid format using Matplotlib."""
	for i, image in enumerate(images):
		plt.subplot(rows, columns, i+1)
		plt.imshow(image, 'gray')
		plt.title(titles[i])
		plt.xticks([]), plt.yticks([])  # Hide tick marks
	plt.show()

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
	
def display_points(in_img, points, radius = 5, colour = (0,0,255)):
    """Draws a circular points on an image"""
    img = in_img.copy()

    if len(colour) == 3:
        if len(img.shape) ==2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for point in points:
        img = cv2.circle(img, tuple(int(x) for x in point), radius, colour, -1)
    show_image(img)
    return img

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

PICTURE_PATH = 'D:/Code/PySem-FInal/PySem-Final/image1.jpg'
img = cv2.imread(PICTURE_PATH, cv2.IMREAD_GRAYSCALE)
processed = pre_process_image(img)
corners = find_corners_of_largest_polygon(processed)
cropped = crop_and_warp(img, corners)
squares = infer_grid(cropped)
digits = get_digits(cropped, squares, 28)

dig = show_digits(digits)
# show_image(dig)

# display_rects(cropped, squares)

# show_image(cropped)

# display_points(processed, corners)

# show_image(processed)


#find the countours in image


ext_contours,new_img  = cv2.findContours(processed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours,new_img  = cv2.findContours(processed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2BGR)



all_contours = cv2.drawContours(processed.copy(), contours, -1, (255,0,0), 2)
external_only = cv2.drawContours(processed.copy(), ext_contours, -1, (255, 0, 0), 2)
# plot_many_images([all_contours, external_only], ['All Contours', 'External Only'])



#binary global threshold using a value of 127
ret, threshold1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

#binary adaptive threshold using 11 nearest neighbor pixels
threshold2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# plot_many_images([threshold1, threshold2], ['Global', 'Adaptive'])

import tensorflow as tf
import pickle
import os
tf.compat.v1.disable_eager_execution()
x = tf.compat.v1.placeholder(tf.float32, shape=[0, 784])  # Placeholder for input
y_ = tf.compat.v1.placeholder(tf.float32, shape=[0, 10])  # Placeholder for true labels (used in training)
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
# More numerically stable to perform Softmax here instead of on the previous layer
# c.f. https://www.tensorflow.org/get_started/mnist/beginners
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Gradient descent and backpropagation learning
train_step = tf.compat.v1.train.GradientDescentOptimizer(0.5).minimize(cost)

# Accuracy comparison/measurement function
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def load_data(file_name):
	"""Loads Python object from disk."""
	with open(file_name, 'rb') as f:
		data = pickle.load(f)
	return data

# Train the network
model_path = 'D:/Code/PySem-FInal/PySem-Final/'
ds = load_data(os.path.join('D:/Code/PySem-FInal/PySem-Final/neural_net', 'digit-basic'))  # Dataset
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, Convolution2D, MaxPooling2D
from keras.utils import np_utils
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
# Import MNIST dataset split into 60,000 for training and 10,000 for testing
dset = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = dset.load_data()
# Insight into imported data
num = 0
# plt.imshow(train_images[num])
# for n in range(5):
#   plt.imshow(train_images[n])
#   print(train_labels[n])

# Categorize labels (not mandatory for our first simple network)
train_labels_cat = np_utils.to_categorical(train_labels, 10)
test_labels_cat = np_utils.to_categorical(test_labels, 10)

# Preprocess image data
test_images_orig = test_images
train_images = train_images / 255
test_images = test_images / 255


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
num = 0                  
predictions = model.predict(resize)


# test_loss, test_acc = model.evaluate(test_images, test_labels_cat)
# print(test_loss,test_acc)

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






digits = '123456789'
rows = 'ABCDEFGHI'
cols = digits

def cross(A, B):
    "Cross product of elements in A and elements in B."
    return [a+b for a in A for b in B]

# How a sudogu grid is represented
# A1 A2 A3| A4 A5 A6| A7 A8 A9  
# B1 B2 B3| B4 B5 B6| B7 B8 B9
# C1 C2 C3| C4 C5 C6| C7 C8 C9
#---------+---------+---------
# D1 D2 D3| D4 D5 D6| D7 D8 D9
# E1 E2 E3| E4 E5 E6| E7 E8 E9
# F1 F2 F3| F4 F5 F6| F7 F8 F9
# ---------+---------+---------
# G1 G2 G3| G4 G5 G6| G7 G8 G9
# H1 H2 H3| H4 H5 H6| H7 H8 H9
# I1 I2 I3| I4 I5 I6| I7 I8 I9

#List comprehension to create all identifiers for each cell in sudoku grid

squares = cross(rows, cols)
#print(squares)
#print('\n')

#Generates all 9 column units
rowUnits = [cross(rows, c) for c in cols]
#print(rowUnits)

#Generates all 9 row units
columnUnits = [cross(r , cols) for r in rows]
#print(columnUnits)

#Generates all 9 box units
boxUnits = [cross(rs, cs) for rs in ('ABC', 'DEF', 'GHI') for cs in ('123','456','789')]
#print(boxUnits)

#Combines all units together
unitList = (rowUnits + columnUnits + boxUnits)


#Makes a dictionary where the specific value 'C1' for example exists in all 3 units
units = dict((s, [u for u in unitList if s in u]) for s in squares)


#Makes all squares in the related 3 units except 'C2' for example
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in squares)


#For now this will be the sudoku grid input -> Later, if there will be extra time, ill make a GUI for this
#This is just easiest for the eyes, to understand how the grid is ploted
gridTest = """
8 . . |. . . |. . . 
. . 3 |6 . . |. . . 
. 7 . |. 9 . |2 . . 
------+------+------
. 5 . |. . 7 |. . . 
. . . |. 4 5 |7 . . 
. . . |1 . . |. 3 . 
------+------+------
. . 1 |. . . |. 6 8 
. . 8 |5 . . |. 1 . 
. 9 . |. . . |4 . . 
"""


def grid_Values(grid):
    "Convert grid into a dictionary of {square: char} with '.' or '0' for ampty cells"
    #Ignores all other symbols that are not 1-9 or .
    chars = [c for c in grid if c in digits or c in '0.']
    assert len(chars) == 81
    return dict(zip(squares, chars))

grid_values = grid_Values(gridTest)
print(grid_values['A1'])

#If a square has only one possible value, then eliminate that value from the squares peers
#If a unit has only one possible place for a value, then put the value there

#Leaves only possible values for each square
def parse_grid(grid):
    """Convert grid to a dict of possible values, {square: digits},
    or
    return False if a contradiction is detected."""
    # To start, every square can be any digit; then asign values from the grid
    values = dict((s, digits) for s in squares)
    for s, d in grid_Values(grid).items():
        if d in digits and not assign(values, s, d):
            return False    #Fail if we can't assign d to square s
    return values

def assign(values, s, d):
    """Eliminate all the other values (except d) from values[s] and propagate
    Return values, exept return False if contradiction is detected"""
    other_values = values[s].replace(d, '')
    if all(eliminate(values, s, d2) for d2 in other_values):
        return values
    else:
        return False


def eliminate(values, s, d):
    """Eliminate d from values[s]; propogate when values or places <= 2
    return values, except return False if a contradiction is detected"""
    if d not in values[s]:
        return values
    values[s] = values[s].replace(d, '')
    # (1) if a square is reduced to one value d2, then eliminate d2 from the peers
    if len(values[s]) == 0:
        return False #Contradiction removed last value
    elif len(values[s]) == 1:
        d2 = values[s]
        if not all(eliminate(values, s2, d2) for s2 in peers[s]):
            return False
    # (2) If a unit u is reduced to only one place for a value d, then put it there
    for u in units[s]:
        dplaces = [s for s in u if d in values[s]]
        if len(dplaces) == 0:
            return False #Contradiction: no place for this value
        elif len(dplaces) == 1: #d can only be in one place in unit; assign it there
            if not assign(values, dplaces[0], d):
                return False
    return values

def display(values):
    "Display these values as a 2-D grid."
    width = 1+max(len(values[s])for s in squares)
    line = '+'.join(['-'*(width*3)]*3)
    for r in rows:
        print(''.join(values[r+c].center(width)+('|'if c in '36' else '')for c in cols))
        if r in 'CF':
            print(line)
    print()

def solve(grid):
    return search(parse_grid(grid))

def search(values):
    "Using depth-first search and propogation, try all possible values"
    if values is False:
        return False #Failed earlier
    if all(len(values[s]) == 1 for s in squares):
        return values #solved
    n, s = min((len(values[s]), s) for s in squares if len(values[s]) > 1)
    return some(search(assign(values.copy(), s, d))for d in values[s])

def some(seq):
    "return some element of seq that is true"
    for e in seq:
        if e: return e
    return False    

display(solve(puzzleString))

