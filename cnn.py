from tensorflow.keras.models import load_model
from tensorflow.keras.layers import LeakyReLU
import tensorflow as tf
import numpy as np
import anvil.server
from tensorflow.keras.utils import register_keras_serializable



# ✅ Register custom Transformer layers
# @register_keras_serializable(package="CustomLayers")
# class PositionalIndex(tf.keras.layers.Layer):
#     def call(self, x):
#         bs = tf.shape(x)[0]  # Extract batch size
#         number_of_vectors = tf.shape(x)[1]  # Count the number of vectors (should be m*n)
#         indices = tf.range(number_of_vectors)  # Index for each vector
#         indices = tf.expand_dims(indices, 0)  # Reshape appropriately
#         return tf.tile(indices, [bs, 1])  # Repeat for each batch

# @register_keras_serializable(package="CustomLayers")
# class ClassTokenIndex(tf.keras.layers.Layer):
#     def call(self, x):
#         bs = tf.shape(x)[0]  # Extract batch size
#         number_of_vectors = 1  # We want just 1 vector for the class token
#         indices = tf.range(number_of_vectors)  # Index for the vector
#         indices = tf.expand_dims(indices, 0)  # Reshape appropriately
#         return tf.tile(indices, [bs, 1])  # Repeat for each batch

# @register_keras_serializable(package="CustomLayers")
# class ClassTokenSelector(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(ClassTokenSelector, self).__init__(**kwargs)

#     def call(self, inputs):
#         """ Extracts the first token (class token) from the sequence. """
#         return inputs[:, 0, :]

#     def get_config(self):
#         config = super(ClassTokenSelector, self).get_config()
#         return config

# # ✅ Paths to the models
# CNN_MODEL_PATH = "/home/bitnami/connect4/cnn_model3.h5"  # Update if necessary
# TRANSFORMER_MODEL_PATH = "/home/bitnami/connect4/transformer_model_final.keras"  # Update if necessary
# ✅ NEW (Correct paths for Docker container)
CNN_MODEL_PATH = "/connect4/cnn_model3.h5"
TRANSFORMER_MODEL_PATH = "/connect4/transformer_model_final.keras"

# # ✅ Load models with custom objects
# custom_objects_cnn = {
#     'LeakyReLU': LeakyReLU
# }

# customer_objects_transformer = {
#     'PositionalIndex': PositionalIndex,
#     'ClassTokenIndex': ClassTokenIndex,
#     'ClassTokenSelector': ClassTokenSelector
# }


# cnn_model = load_model(CNN_MODEL_PATH, custom_objects=custom_objects_cnn)
# transformer_model = load_model(TRANSFORMER_MODEL_PATH, custom_objects=customer_objects_transformer)
# cnn_model = load_model(CNN_MODEL_PATH, custom_objects=custom_objects_cnn, compile=False)
# transformer_model = load_model(TRANSFORMER_MODEL_PATH, custom_objects=customer_objects_transformer, compile=False)
cnn_model = load_model(CNN_MODEL_PATH)
transformer_model = load_model(TRANSFORMER_MODEL_PATH)



def swap_board(board):
    """
    Swap [0,1] to [1,0] and [1,0] to [0,1] in a given board.
    Ensures AI always predicts for the +1 player.
    """
    board = np.array(board)  # Ensure it's a NumPy array

    # Create masks
    mask_01 = (board[:, :, 0] == 0) & (board[:, :, 1] == 1)  # Find [0,1]
    mask_10 = (board[:, :, 0] == 1) & (board[:, :, 1] == 0)  # Find [1,0]

    # Swap values
    board[mask_01] = [1, 0]
    board[mask_10] = [0, 1]

    return board  # Return the modified board

@anvil.server.callable
def get_best_move(board_list, is_user_first, model_type="cnn"):
    """
    Receives board state from Anvil, processes it, and returns the best move.
    Allows the user to choose between CNN or Transformer.
    
    Parameters:
        board_list (list): The 6x7x2 board state.
        is_user_first (bool): Whether the user played first.
        model_type (str): "cnn" or "transformer", determines which model to use.
    """
    # Convert board from list to NumPy array
    board = np.array(board_list)

    # Flip board state if user went first
    if is_user_first:  # if user is plus, CNN needs to be minus
        board = swap_board(board)

    # CNN Model Processing
    if model_type == "cnn":
        input_board = board.reshape(1, 6, 7, 2)  # Keep 6x7x2 shape
        predictions = cnn_model.predict(input_board)[0]
        best_move = int(np.argmax(predictions))

    # Transformer Model Processing
    elif model_type == "transformer":
        input_board = board.reshape(1, 42, 2)  # Reshape into (1, 42, 2)
        predictions = transformer_model.predict(input_board)[0]
        best_move = int(np.argmax(predictions))  # Select move with highest probability

    else:
        raise ValueError("Invalid model type. Choose 'cnn' or 'transformer'.")

    return best_move

# Keep the server running on AWS
anvil.server.wait_forever()
