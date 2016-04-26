import numpy as np

class Connect4():

    def __init__(self):

        # Game state is represented by a matrix of zeroes, ones, and two
        # 0 represents an open space
        # 1 represents player one's piece
        # 2 represents player two's piece
        self.state = np.zeros((6, 7), dtype=0)

        # Game starts with player 1 turn
        self.turn = 1