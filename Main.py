from Solver import Solver
from Reader import Reader
from GUI import Game
readingPart = Reader()
puzzleArray = readingPart.Read('Untitled-1_2260717b.jpg')
solvingPart = Solver()
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
solveArray = solvingPart.Solve(str(puzzleString))

print(solveArray)


gamePart = Game()
gamePart.StartTheGame(puzzleArray, solveArray)
