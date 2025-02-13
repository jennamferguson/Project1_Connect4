import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


MODEL_PATH = "/FOLDERNAME/tensorflow_model.h5"  # Update with the actual path
model = tf.keras.models.load_model(MODEL_PATH)


@anvil.server.callable
def get_best_move(board, model):

    board_input = board[np.newaxis, ...]  

    # Get the model's prediction
    prediction = model.predict(board_input)

    # find column with the highest predicted value
    best_move = np.argmax(prediction)

    return best_move


# Keep the server running on AWS
anvil.server.wait_forever()

