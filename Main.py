from Solver import Solver   #Imports the sudoku solving algorithm
from Reader import Reader   #Imports the sudoku reader part
from GUI import Game        #Imports the sudoku game part

readingPart = Reader()
puzzleArray = readingPart.Read('sudoku-puzzle-games.jpg')

#Gets a 2D list with the sudoku grid (not solwed)
solvingPart = Solver()

#Converts the array to string
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

#Gets solves previously gotten sudoku problem (solves sudoku 2D list)
solveArray = solvingPart.Solve(str(puzzleString))

#Starts the game
gamePart = Game()
gamePart.StartTheGame(puzzleArray, solveArray)
