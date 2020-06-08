from Solver import Solver
from Reader import Reader
from GUI import Grid
readingPart = Reader()
puzzleArray = readingPart.Read('Untitled-1_2260717b.jpg')
gamePart = Grid()
gamePart.OpenSolver(puzzleArray)
# print(puzzleString)

# solvingPart = Solver()
# solvingPart.Solve(str(puzzleString))