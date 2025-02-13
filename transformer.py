import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


MODEL_PATH = "/FOLDERNAME/transformer_model_final.keras"  # Update with the actual path
model = tf.keras.models.load_model(MODEL_PATH)


@anvil.server.callable
def select_best_move(transformer_prediction, board):
    """
    Selects the best move by considering both the model's prediction and defensive plays.

    - If an opponent is about to win, block them.
    - Otherwise, choose the model's top prediction.
    """
    predicted_move = np.argmax(transformer_prediction)  # Model's top choice

    # Check if opponent is about to win and block them
    for col in range(7):  # Iterate through all possible columns
        temp_board = update_board(board.copy(), "minus", col)  # Simulate opponent move
        if check_for_win(temp_board, col) != "nobody":  # If opponent wins, block
            return col  # Prioritize blocking

    return predicted_move  # Default to model’s choice if no block is needed

# Keep the server running on AWS
anvil.server.wait_forever()

