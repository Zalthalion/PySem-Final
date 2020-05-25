# pip install opencv-python
import cv2
# pip install keras
# pip install jupyter
# pip install sckit-learn

def pre_process_image(img, skip_dilate = False):
    "Uses a blurring function, adaptive thresholding and dilation to expose the main features"

    #gausian blur with kernel size (height, width) of 9
    #note that kernel sizes must be positive and odd and the kernel must be square
    proc = cv2.GaussianBlur(img.copy(), (9,9),0)
    #adaptive threshold using 11 nearest neighbor pixels

    proc = cv2.adaptiveThreshold(proc, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11,2)

    #Invert colors, so gridlines have non-zero pixel values
    #Necessary to dilaate the image, otherwise will look erosion instead

    if not skip_dilate:
        #Dilate the image to increase the size of the gridlines
        kernel = np.array([[0.,1.,0.,], [1., 1., 1.], [0., 1., 0.,]], np.unit8)
        proc = cv2.dilate(proc, kernel)

    plt.imshow(proc, cmap='gray')
    plt.title('pre_process_image')
    plt.show()
    return proc

def find_corners_of_largest_polygon(img):
    "Finds the 4 extreme corners of the largest countour in the image"
    _, contours, h = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #Find contours
    contours = sorted(contours, key = cv2.contourArea, reverse = True) #sort area, descending
    polygon = contours[0] #largest image

    #use of'operator.itemgetter' with 'max' and 'min' allows us to get the index of the point
    #each point is an array of 1 coordinate, hence the [0] getter, then [0] or [1] used to get x and y respectively

    #Bottom-right point has the largest (x+y) value
    #Top left has point smallest(x+y)value
    #Bottom-left point has smallest (x-y) value
    #top right has largest x-y value
    bottom_right, _ = max(enumerate(pt[0][0] + pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    top_left, _ = min(enumerate(pt[0][0] + pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    bottom_left, _ = min(enumerate(pt[0][0] - pt[0][1] for pt in polygon), key=operator.itemgetter(1))
    top_right, _ = max(enumerate(pt[0][0] - pt[0][1] for pt in polygon), key=operator.itemgetter(1))

    #return an array of all 4 points using the indicies'
    #each point is in its own array of the coordinate

    return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def crop_and_warp(img, crop_rect):
    "crops and warps a rectangular section form an image into a square of similar size"

    #rectangle described by top left, top right, bottom right and bottom left points
    top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]

    #Explicity set the data type to float32 or getPerspectiveTransform will throw an error
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

    #get the longest sode in the rectangle
    side = max([
        distance_between(bottom_right, top_right),
        distance_between(top_left, bottom_left),
        distance_between(bottom_right, bottom_left),
        distance_between(top_left, top_right)
        ])

    #Describe a square with side of the calculated length, this is new perspective we want
    sdt = np.array([[0,0], [side-1,0], [side - 1, side -1], [0, side-1]], dtype = 'float32')

    #Gets the transformation matrix for skewing the image to fit a square by comparing the 4 before and after points
    m = csv2.getPerspectiveTransforms(src, dst)

    #performs the transformations on the original image
    warp = cv2.wapPerspective(img, m, (int(side), int(side)))
    plt.imshow(warp, cmap='gray')
    plt.title('warp_image')
    plt.show()
    return warp

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Canv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from keras.utils import np_utils

#load dataset
data = pd.read_csv('image_data.csv')

#split into input and output variables
X = []
Y = data['y']
del data['y']

for i in range(data.shape[0]):
    flat_pixels = data.iloc[i].values[1:]
    image = np.reshape(flat_pixels, (28,28))
    X.append(image)

X = np.array(X)
Y = np.Array(Y)







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
print(unitList)
print(len(unitList))

#Makes a dictionary where the specific value 'C1' for example exists in all 3 units
units = dict((s, [u for u in unitList if s in u]) for s in squares)
print(units['C2']) 

#Makes all squares in the related 3 units except 'C2' for example
peers = dict((s, set(sum(units[s],[]))-set([s])) for s in squares)
print(peers['C2'])

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

display(solve(gridTest))

