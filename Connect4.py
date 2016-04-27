import random

class Connect4():

    def __init__(self):

        # Game state is represented by a matrix of zeroes, ones, and two
        # 0 represents an open space
        # 1 represents player one's piece
        # 2 represents player two's piece
        self.board = [[0 for x in range(7)] for y in range(6)]

        # Game starts with player 1 turn
        self.turn = self.whoGoesFirst()

        #
        self.win = 0

    def changeTurn(self):
        self.turn = 1 if self.turn == 2 else 2

    def move(self, player, column):
        if (0 <= column <= 6 and self.board[5][column] != 0):
            print("Can't move here, please enter a valid position")
            return
        else: # valid move, place piece at first available row in column
            for row in range(6):
                if self.board[row][column] == 0:
                    self.board[row][column] = self.turn
                    self.checkForWin(player, row, column)
                    self.changeTurn()
                    break

    # Given a player and the row/column they put their piece, check if that move made them win
    def checkForWin(self, player, row, column):
        if self.checkVertical(player, row, column) or self.checkHorizontal(player, row, column) or self.checkDiagonals(player, row, column):
            print("----------------------------")
            print("Yay! Player %i is the winner!" % player)
            print("----------------------------")
            self.win = player

    def checkVertical(self, player, row, column):
        count = 0
        for x in range(-3, 4):
            if 0 <= (row + x) <= 5 and self.board[row+x][column] == player:
                count+=1
                if count == 4:
                    return True
            else:
                count = 0
        return False

    def checkHorizontal(self, player, row, column):
        count = 0
        for x in range(-3, 4):
            if (0 <= (column + x) <= 6 and self.board[row][column+x] == player):
                count+=1
                if count == 4:
                    return True
            else:
                count = 0
        return False

    def checkDiagonals(self, player, row, column):
        count = 0
        for x in range(-3, 4):
            if 0 <= (column + x) <= 6 and 0 <= (row - x) <= 5 and self.board[row-x][column+x] == player:
                count+=1
                if count == 4:
                    return True
            else:
                count = 0

        count = 0
        for x in range(-3, 4):
            if 0 <= (column + x) <= 6 and 0 <= (row + x) <= 5 and self.board[row+x][column+x] == player:
                count+=1
                if count == 4:
                    return True
            else:
                count = 0
        return False

    def whoGoesFirst(self):
        return random.randint(1, 2)

    def restartGame(self):
        self.board = [[0 for x in range(7)] for y in range(6)]
        self.turn = self.whoGoesFirst()
        self.win = 0

    def printBoard(self):
        key = [" ", "X", "O"]
        print(" ")
        for row in range(5, -1, -1):
            print(key[self.board[row][0]]+" "+key[self.board[row][1]]+" "+key[self.board[row][2]]+" "+key[self.board[row][3]]+" "+key[self.board[row][4]]+" "+key[self.board[row][5]]+" "+key[self.board[row][6]])
        print("1 2 3 4 5 6 7")

if __name__ == "__main__":
    Game = Connect4()
    while True:
        print("Player%i, take your move." % Game.turn)
        col = int(input("Enter col of move..."))
        Game.move(Game.turn, (col-1))
        Game.printBoard()
        if Game.win:
            restart = int(input("Enter 1 to restart the game..."))
            if restart == 1:
                Game.restartGame()
            else:
                break