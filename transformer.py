import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the Transformer model instead of CNN
transformer_model = load_model('transformer.h5')

anvil.server.connect("your-anvil-uplink-key")  # Replace with your actual Anvil key

def update_board(board_temp, color, column):
    """
    Updates the board (6x7x2) by placing a checker in the specified column.
    """
    board = board_temp.copy()
    
    for row in range(5, -1, -1):  # Start from bottom row
        if board[row, column, 0] == 0 and board[row, column, 1] == 0:
            if color == 'plus':
                board[row, column, 0] = 1
                board[row, column, 1] = 0
            else:
                board[row, column, 0] = 0
                board[row, column, 1] = 1
            return board
    
    return board  # Return unchanged if column is full

def swap_board(board):
    """
    Swap [0,1] to [1,0] and [1,0] to [0,1] in a given board only if player == -1.
    """
    board = np.array(board)  
    mask_01 = (board[:,:,0] == 0) & (board[:,:,1] == 1)
    mask_10 = (board[:,:,0] == 1) & (board[:,:,1] == 0)
    board[mask_01] = [1, 0]
    board[mask_10] = [0, 1]
    return board

def check_for_win(board, col):
    """
    Checks for a win in the 6x7x2 board.
    """
    nrow, ncol = 6, 7
    
    for row in range(6):
        if board[row, col, 0] == 1 or board[row, col, 1] == 1:
            break
    
    player = "plus" if board[row, col, 0] == 1 else "minus"
    check_channel = 0 if player == "plus" else 1
    
    def check_direction(delta_row, delta_col):
        count = 0
        r, c = row, col
        while 0 <= r < nrow and 0 <= c < ncol and board[r, c, check_channel] == 1:
            count += 1
            r += delta_row
            c += delta_col
        return count

    if check_direction(1, 0) >= 4:
        return f'v-{player}'
    left_count = check_direction(0, -1)
    right_count = check_direction(0, 1)

    if left_count + right_count - 1 >= 4:
        return f'h-{player}'
    down_right = check_direction(1, 1)
    up_left = check_direction(-1, -1)

    if down_right + up_left - 1 >= 4:
        return f'd-{player}'
    down_left = check_direction(1, -1)
    up_right = check_direction(-1, 1)
    
    if down_left + up_right - 1 >= 4:
        return f'd-{player}'
    return 'nobody'

@anvil.server.callable
def get_recommended_move(board, user_player):
    """
    Uses the Transformer model to return the best move.
    """
    board = np.array(board)
    if board.shape != (6,7,2):
        raise ValueError("Invalid board shape. Expected (6,7,2).")
    
    # Flatten board for Transformer input
    board_flat = board.reshape(42, 2)  # Transformer expects (42, 2)
    
    # Swap board if player is 'minus'
    if user_player == 'minus':
        swapped_board = swap_board(board)
        board_flat = swapped_board.reshape(42, 2)
    
    # Make prediction using the Transformer model
    transformer_prediction = transformer_model.predict(board_flat[np.newaxis, ...])
    
    return int(np.argmax(transformer_prediction))  # Return best move

# Keep the server running on AWS
anvil.server.wait_forever()
