from Solver import Solver
from Reader import Reader
from GUI import Game
readingPart = Reader()
puzzleArray = readingPart.Read('Untitled-1_2260717b.jpg')
gamePart = Game()
gamePart.StartTheGame(puzzleArray)
# print(puzzleString)

# solvingPart = Solver()
# solvingPart.Solve(str(puzzleString))