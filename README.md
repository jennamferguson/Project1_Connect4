 # AI vs. Human: A Deep Learning Approach to Connect 4 

This project involves training a neural network to play Connect 4 using Monte Carlo Tree Search (MCTS) to generate a dataset of board positions and optimal moves. Two deep learning models—a Convolutional Neural Network (CNN) and a Transformer—are trained on this dataset to predict the best move for a given board state. The project also includes a web interface where users can play against these models.

Play against our models here: https://flat-pale-entry.anvil.app (Username: dan , Password: Optimization1234) - Best of luck!

## Dataset Generation
Uses Monte Carlo Tree Search (MCTS) to play games against itself and recommend the best moves.
The dataset consists of board positions (X) and corresponding best moves (Y).
Some randomness is introduced in the opening moves to diversify the dataset.
Board representation is A 6×7×2 numpy array. We found this to be best for learning efficiency. 

The dataset can be found in gitignore

## Model Training

### Neural Network Architectures:
#### CNN: 
Uses convolutional layers to process the board as an image.
#### Transformer:
Processes board positions using multi-head self-attention mechanisms.
#### Evaluation:
Models are tested against an MCTS bot to measure win rates.
Performance metrics include accuracy on validation boards and number of moves to win/lose.

## Play Game
The resulting models as well as a notebook to play a game of Connect4 against these models is included in the Play Game folder. 

cnn_model.h5 and cnn_model2.h5 are earlier models. While these models are competitive opponents, we found they lacked the ability to block early wins. cnn_model3.h5 is our strongest player throughout the game - we recommend playing this if you're up for the challenge! This model had a validation accuracy of 91% 

## Backend Code
This folder includes the code the backend code to host these models on Anvil
