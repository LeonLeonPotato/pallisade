import numpy as np

def get_possible_actions(state : np.ndarray):
    # actions = []
    # for x in range(0, 3):
    #     for y in range(0, 3):
    #         if state[x, y] == 0:
    #             actions.append((x, y))
    # return actions

    return np.array(np.where(state == 0)).T

def check_win2(board):
    # Check rows
    for row in board:
        if np.all(row == 1):
            return 1
        elif np.all(row == -1):
            return -1
    # Check columns
    for col in board.T:
        if np.all(col == 1):
            return 1
        elif np.all(col == -1):
            return -1
    # Check diagonals
    if np.all(np.diag(board) == 1) or np.all(np.diag(np.fliplr(board)) == 1):
        return 1
    elif np.all(np.diag(board) == -1) or np.all(np.diag(np.fliplr(board)) == -1):
        return -1
    # If no winner found, return None
    if np.count_nonzero(board == 0) == 0:
        return 2
    return 0

import numpy as np

def check_win(board):
    """
    Takes in a 7x7 numpy array with elements {-1, 0, 1} and returns the winner
    of the connect 4 game. -1 means player 2's tile, 1 means player 1's tile.
    0 means unoccupied. Returns 1 if player 1 won, -1 if player 2 won,
    0 if the game is still going on, and 2 if the game is a draw.
    """
    # Check rows
    for i in range(7):
        for j in range(4):
            if board[i][j] != 0 and board[i][j] == board[i][j+1] == board[i][j+2] == board[i][j+3]:
                return board[i][j]

    # Check columns
    for i in range(4):
        for j in range(7):
            if board[i][j] != 0 and board[i][j] == board[i+1][j] == board[i+2][j] == board[i+3][j]:
                return board[i][j]

    # Check diagonals
    for i in range(4):
        for j in range(4):
            if board[i][j] != 0 and board[i][j] == board[i+1][j+1] == board[i+2][j+2] == board[i+3][j+3]:
                return board[i][j]
            if board[i][j+3] != 0 and board[i][j+3] == board[i+1][j+2] == board[i+2][j+1] == board[i+3][j]:
                return board[i][j+3]

    # Check if the game is still going on
    if np.any(board == 0):
        return 0

    # If no winner and no empty cells, it's a draw
    return 2

