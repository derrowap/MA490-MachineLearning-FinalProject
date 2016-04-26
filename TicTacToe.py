# import numpy as np
import random

class TicTacToe():

	def __init__(self):
		# State is a 3x3 matrix representation of the current game
		# A game that looks like:
		# X |   | O
		# ---------
		# O |   | X
		# ---------
		#   | X | O
		# The matrix will be:
		# [ [1, 0, 2], [2, 0, 1], [0, 1, 2] ]
		# Such that empty space is 0, 'X' is 1, and 'O' is 2.
		self.state = [[0 for x in range(3)] for y in range(3)]
		# If turn is 1, it is Player1's turn.
		# If turn is 2, it is Player2's turn.
		self.turn = self.whoGoesFirst()
		# If win is 0, nobody wins.
		# If win is 1, Player1 wins.
		# If win is 2, Player2 wins.
		self.win = 0

	def changeTurn(self):
		if self.turn == 1:
			self.turn = 2
		else:
			self.turn = 1

	def move(self, player, row, col):
		if player != self.turn: # not this player's turn
			print("Oops! It is not your turn, Player%i" % player)
			return
		if self.state[row][col]: # already a move in this position
			print("Oops! It seems that someone has already moved here, Player%i" % player)
			return
		else: # empty spot
			self.state[row][col] = player
			self.checkForWin(self.state, player)
			self.changeTurn()

	# Given a player, checks all possibilities to see if they win
	def checkForWin(self, board, player):
		if ((board[0][0] == player and board[0][1] == player and board[0][2] == player) or
			(board[1][0] == player and board[1][1] == player and board[1][2] == player) or
			(board[2][0] == player and board[2][1] == player and board[2][2] == player) or
			(board[0][0] == player and board[1][1] == player and board[2][2] == player) or
			(board[0][2] == player and board[1][1] == player and board[2][0] == player) or
			(board[0][0] == player and board[1][0] == player and board[2][0] == player) or
			(board[0][1] == player and board[1][1] == player and board[2][1] == player) or
			(board[0][2] == player and board[1][2] == player and board[2][2] == player)):
			print("Yay! Player%i is the winner!" % player)
			self.win = player

	def whoGoesFirst(self):
		return random.randint(1, 2)

	def restartGame(self):
		self.state = [[0 for x in range(3)] for y in range(3)]
		self.turn = self.whoGoesFirst()
		self.win = 0

	def printBoard(self):
		key = [' ', 'X', 'O']
		print('   |   |')
		print(' ' + key[self.state[0][0]] + ' | ' + key[self.state[0][1]] + ' | ' + key[self.state[0][2]])
		print('   |   |')
		print('-----------')
		print('   |   |')
		print(' ' + key[self.state[1][0]] + ' | ' + key[self.state[1][1]] + ' | ' + key[self.state[1][2]])
		print('   |   |')
		print('-----------')
		print('   |   |')
		print(' ' + key[self.state[2][0]] + ' | ' + key[self.state[2][1]] + ' | ' + key[self.state[2][2]])
		print('   |   |')

if __name__ == "__main__":
	Game = TicTacToe()
	while True:
		print("Player%i, take your move." % Game.turn)
		row = int(raw_input("Enter row of move..."))
		col = int(raw_input("Enter col of move..."))
		Game.move(Game.turn, row, col)
		Game.printBoard()
		if Game.win:
			restart = int(raw_input("Enter 1 to restart the game"))
			if restart == 1:
				Game.restartGame()
			else:
				return