from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import numpy as np
import anvil.server

# Path to your model
# MODEL_PATH = "/home/bitnami/connect4/cnn_model3.h5"    # Update if necessary
MODEL_PATH = "/connect4/cnn_model3.h5"

# Custom object dictionary for loading the model
custom_objects = {'LeakyReLU': LeakyReLU}

# Load the model with custom objects
model = load_model(MODEL_PATH, custom_objects=custom_objects)

@anvil.server.callable
def get_best_move(board_list, is_user_first):
    """
    Receives board state from Anvil, processes it, and returns the best column for the move.
    Ensures CNN always predicts for +1 by flipping the board if the user played first.
    """
    # Convert board from JSON list to NumPy array
    board = np.array(board_list)

    # Flip board if the user played first
    if is_user_first:
        board *= -1  # Swap board perspective

    # Ensure correct input shape for CNN
    input_board = board.reshape(1, 6, 7, 2)

    # Get model prediction
    predictions = model.predict(input_board)[0]
    best_move = int(np.argmax(predictions))  # Choose the best column

    return best_move



# Keep the server running on AWS
anvil.server.wait_forever()

