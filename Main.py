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
4 . . |. . . |8 . 5 
. 3 . |. . . |. . . 
. . . |7 . . |. . . 
------+------+------
. 2 . |. . . |. 6 . 
. . . |. 8 . |4 . . 
. . . |. 1 . |. . . 
------+------+------
. . . |6 . 3 |. 7 . 
5 . . |2 . . |. . . 
1 . 4 |. . . |. . . 
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
    for s, d in grid_values(grid).items():
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
