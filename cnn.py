import anvil.server
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model


cnn_model = load_model('cnn_model1.h5')
anvil.server.connect("your-anvil-uplink-key")  # Replace with your actual Anvil key


def update_board(board_temp, color, column):
    """
    Updates the board (6x7x2) by placing a checker in the specified column.
    
    Parameters:
    - board_temp: 6x7x2 NumPy array representing the board
    - color: 'plus' for +1, 'minus' for -1
    - column: Integer (0-6) representing the column where the piece is dropped
    
    Returns:
    - Updated 6x7x2 board with the new piece added
    """
    board = board_temp.copy()
    
    # Find the lowest available row in the given column
    for row in range(5, -1, -1):  # Start from bottom row
        if board[row, column, 0] == 0 and board[row, column, 1] == 0:  # Check if empty
            if color == 'plus':
                board[row, column, 0] = 1  # Set +1 in first channel
                board[row, column, 1] = 0
            else:
                board[row, column, 0] = 0
                board[row, column, 1] = 1  # Set -1 in second channel
            return board  # Return updated board
    
    # If column is full, return the board unchanged
    return board

def swap_board(board):
    """
    Swap [0,1] to [1,0] and [1,0] to [0,1] in a given board only if player == -1.
    """

    board = np.array(board)  

    # Create masks
    mask_01 = (board[:,:,0] == 0) & (board[:,:,1] == 1)  # Find [0,1]
    mask_10 = (board[:,:,0] == 1) & (board[:,:,1] == 0)  # Find [1,0]

    # Swap values
    board[mask_01] = [1, 0]
    board[mask_10] = [0, 1]

    return board  # Return the modified or original board


def check_for_win(board, col):
    """
    Checks for a win in the 6x7x2 board.

    Parameters:
    - board: 6x7x2 NumPy array
    - col: Integer (0-6) indicating the last column where a checker was dropped

    Returns:
    - 'v-plus', 'v-minus' for vertical win
    - 'h-plus', 'h-minus' for horizontal win
    - 'd-plus', 'd-minus' for diagonal win
    - 'nobody' if no win
    """
    nrow, ncol = 6, 7
    
    # Find the row of the last played move
    for row in range(6):
        if board[row, col, 0] == 1 or board[row, col, 1] == 1:
            break  # Found the last placed checker
    
    # Identify which player made the move
    player = "plus" if board[row, col, 0] == 1 else "minus"
    check_channel = 0 if player == "plus" else 1
    
    def check_direction(delta_row, delta_col):
        """Counts consecutive checkers in a specific direction"""
        count = 0
        r, c = row, col
        while 0 <= r < nrow and 0 <= c < ncol and board[r, c, check_channel] == 1:
            count += 1
            r += delta_row
            c += delta_col
        return count

    # **Check vertical (↓)**
    if check_direction(1, 0) >= 4:
        return f'v-{player}'
    
    # **Check horizontal (← →)**
    left_count = check_direction(0, -1)  # Count leftwards
    right_count = check_direction(0, 1)  # Count rightwards
    if left_count + right_count - 1 >= 4:
        return f'h-{player}'

    # **Check diagonal (↘ ↖)**
    down_right = check_direction(1, 1)
    up_left = check_direction(-1, -1)
    if down_right + up_left - 1 >= 4:
        return f'd-{player}'

    # **Check diagonal (↙ ↗)**
    down_left = check_direction(1, -1)
    up_right = check_direction(-1, 1)
    if down_left + up_right - 1 >= 4:
        return f'd-{player}'

    return 'nobody'  # No win detected


@anvil.server.callable
def get_recommended_move(board, user_player):
    """
    Uses the CNN model to return the best move.
    
    Parameters:
    - board: 6x7x2 NumPy array representing the current board state.
    - user_player: 'plus' or 'minus' (indicating whose turn it is).

    Returns:
    - Integer (0-6) representing the recommended column for the move.
    """
    # Ensure board is a valid NumPy array
    board = np.array(board)
    if board.shape != (6,7,2):
        raise ValueError("Invalid board shape. Expected (6,7,2).")

    # Use the CNN model to predict the best move
    if user_player == 'minus':
        swapped_board = swap_board(board)  # Flip board for the model
        cnn_prediction = cnn_model.predict(swapped_board[np.newaxis, ...])
    else:
        cnn_prediction = cnn_model.predict(board[np.newaxis, ...])

    return int(np.argmax(cnn_prediction))  # Return best move from CNN


# Keep the server running on AWS
anvil.server.wait_forever()

