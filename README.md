# MA490-MachineLearning-FinalProject
<p>
Here will be the codebase for our Final Project in MA490 Machine Learning.

<br>As a general overview, we are using a neural network from TensorFlow in Python to learn how to play the game Connect 4.
Once the Neural Net has trained enough against itself, it will be available to play against by utilizing Firebase.
Firebase will host all the data we need for the front and back end to interact with it.
The playable version will be hosted somewhere on my website http://derrow-pinion.com.
This will allow for us to see patterns and algorithms that the Nueral Net learned in order to win.

<br><br>After the Connect 4 version is finished, we wish to move on to train a different Neural Net on how to play Texas Hold'Em.
The Neural Net will play against itself to train. 
It is somewhat ambitious to state that there will be a working, online, playable game against this Neural Network for Texas Hold'Em.
But if it does become available, this would be very useful in being able to experience first hand what algorithms and strategies the
Neural Net has learned.

<br><br>After all of this is accomplished, we will write a formal paper that reflects on the results of this project.

</p>
<h4>Here is the current thought process of how the Connect 4 game will interact with Firebase, HTML client, and Python client:</h4>
<pre>
HTML client:
	Gets matrix from firebase
		Updates graphics
	Get leaderboard from firebase
		Update graphics
	Get boolean tuple who wins (UserWins, NeuralNetWins) from firebase
	If UserWins || NeuralNetWins:
		Update Graphics to display that player wins
	If restart game button pressed:
		Update matrix on firebase to empty
		Set who's turn it is (who won last)
	Get who's turn it is from firebase
	If User's turn:
		Show circle at top of connect 4 as user hovers over each column
		When user clicks while hovering over column:
			Update matrix on firebase
			Update turn to Neural Net's turn on firebase
			(python code checks to see if there is winner)
	If Neural Net's turn:
		Wait until user's turn

Python client:
	Get who's turn from firebase
	If User's turn:
		Wait until NeuralNet's turn
	If NeuralNet's Turn:
		Get matrix from firebase
		Check for User win in matrix
		If User win:
			Update win tuple in firebase
			Update leaderboard in firebase (User++)
		Else:
			Send NeuralNet matrix of game
			Recieve prediction from NeuralNet
			Update matrix from NeuralNet prediction
		Check for NeuralNetWin in matrix
		If NeuralNetWins:
			Update win tuple in firebase
			Update leaderboard in firebase (NeuralNet++)
	Wait for NeuralNet's turn:
		if someone won the game, this is a wait for user to restart game
</pre>
