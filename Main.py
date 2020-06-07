from Solver import Solver
from Reader import Reader
readingPart = Reader()
puzzleString = readingPart.Read('D:/Code/PySem-FInal/PySem-Final/sudoku-puzzle1.jpg')
print(puzzleString)

solvingPart = Solver()
solvingPart.Solve(str(puzzleString))