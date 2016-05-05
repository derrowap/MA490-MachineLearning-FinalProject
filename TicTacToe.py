import random

def main():
	"""Begins a game of tic-tac-toe."""
	Game = TicTacToe()
	print("Welcome to Tic-Tac-Toe")
	while True:
		print("Player%d, take your move." % Game.turn)
		row = int(input("Enter row of move... "))
		col = int(input("Enter col of move... "))
		Game.move(Game.turn, row, col)
		Game.printBoard()
		if Game.win:
			restart = int(input("Enter 1 to restart the game, 0 to end game... "))
			if restart == 1:
				Game.restartGame()
			else:
				print("Closing Tic-Tac-Toe Game...")
				return

class TicTacToe():
	"""A tic-tac-toe game. 

	Two players take turns specifying the row and column to make
	their next move. When a player wins, they can either exit or restart.

	Attributes
	----------
	state: 3x3 integer array
		Current state of the game where 0 is empty, 1 for X, and 2 for O.
	turn: integer
		Represents whos turn it currently is. 1 for X and 2 for O.
	win: integer
		0 if there is no current winner. 1 if X wins, 2 if O wins.
	"""
	def __init__(self):
		self.state = [[0 for x in range(3)] for y in range(3)]
		self.turn = self.whoGoesFirst()
		self.win = 0

	def changeTurn(self):
		"""Flips who turn it is in this game."""
		if self.turn == 1:
			self.turn = 2
		else:
			self.turn = 1

	def move(self, player, row, col):
		"""Moves the player at the given row and col in the board.

		If that position has already been taken, will alert player and 
		it will still be their turn.
		"""
		if player != self.turn: # not this player's turn
			print("Oops! It is not your turn, Player%d" % player)
			return
		if self.state[row][col]: # already a move in this position
			print("Oops! It seems that someone has already moved here, Player%d" % player)
			return
		else: # empty spot
			self.state[row][col] = player
			self.checkForWin(self.state, player)
			self.changeTurn()

	def checkForWin(self, board, player):
		"""Checks all possibilities for player to win."""
		if ((board[0][0] == player and board[0][1] == player and board[0][2] == player) or
			(board[1][0] == player and board[1][1] == player and board[1][2] == player) or
			(board[2][0] == player and board[2][1] == player and board[2][2] == player) or
			(board[0][0] == player and board[1][1] == player and board[2][2] == player) or
			(board[0][2] == player and board[1][1] == player and board[2][0] == player) or
			(board[0][0] == player and board[1][0] == player and board[2][0] == player) or
			(board[0][1] == player and board[1][1] == player and board[2][1] == player) or
			(board[0][2] == player and board[1][2] == player and board[2][2] == player)):
			print("----------------------------")
			print("Yay! Player%d is the winner!" % player)
			print("----------------------------")
			self.win = player

	def whoGoesFirst(self):
		"""Assigns what player goes first in this game.

		Randomly chooses either 1 or 2, representing which player.
		"""
		return random.randint(1, 2)

	def restartGame(self):
		"""Restarts this game's values to a new game."""
		self.state = [[0 for x in range(3)] for y in range(3)]
		self.turn = self.whoGoesFirst()
		self.win = 0

	def printBoard(self):
		"""Prints the current state of the tic-tac-toe board."""
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
	main()		