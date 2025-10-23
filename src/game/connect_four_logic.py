# --- Constants ---
ROW_COUNT = 6
COLUMN_COUNT = 7

# --- Game State Functions ---


def create_board():
    """
    Initializes the 6x7 board with zeros.
    The board is indexed from bottom-up: board[0][c] is the bottom row.
    0: Empty, 1: Player 1 (Red), 2: Player 2 (Yellow)
    """
    board = [[0] * COLUMN_COUNT for _ in range(ROW_COUNT)]
    return board


def drop_piece(board, row, col, piece):
    """
    Updates the board state by dropping a piece into the specified (row, col).
    Assumes the move has already been validated.
    """
    board[row][col] = piece


def is_valid_location(board, col):
    """
    Checks if the top slot of a column is empty, meaning a piece can be dropped.
    The top slot is at index ROW_COUNT - 1.
    """
    if 0 <= col < COLUMN_COUNT:
        return board[ROW_COUNT - 1][col] == 0
    return False


def get_next_open_row(board, col):
    """
    Finds the lowest empty row index (r) in a given column (col).
    Returns -1 if the column is full.
    """
    for r in range(ROW_COUNT):
        if board[r][col] == 0:
            return r
    return -1  # Column is full


def check_win(board, piece):
    """
    Checks the entire board for 4-in-a-row for the specified piece (1 or 2).
    Checks horizontally, vertically, and both diagonal directions.
    """
    # 1. Check horizontal locations
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT):
            if (
                board[r][c] == piece
                and board[r][c + 1] == piece
                and board[r][c + 2] == piece
                and board[r][c + 3] == piece
            ):
                return True

    # 2. Check vertical locations (only need to check up to ROW_COUNT - 3)
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c] == piece
                and board[r + 2][c] == piece
                and board[r + 3][c] == piece
            ):
                return True

    # 3. Check positive slope diagonals (bottom-left to top-right)
    # Start checking from the bottom 3 rows and left 4 columns
    for c in range(COLUMN_COUNT - 3):
        for r in range(ROW_COUNT - 3):
            if (
                board[r][c] == piece
                and board[r + 1][c + 1] == piece
                and board[r + 2][c + 2] == piece
                and board[r + 3][c + 3] == piece
            ):
                return True

    # 4. Check negative slope diagonals (top-left to bottom-right)
    # Start checking from the top 3 rows and left 4 columns
    for c in range(COLUMN_COUNT - 3):
        for r in range(3, ROW_COUNT):  # Start from row 3 (0-indexed) up to 5
            if (
                board[r][c] == piece
                and board[r - 1][c + 1] == piece
                and board[r - 2][c + 2] == piece
                and board[r - 3][c + 3] == piece
            ):
                return True

    return False


def is_board_full(board):
    """
    Checks if there are any valid moves remaining (i.e., if the top row is full).
    """
    for c in range(COLUMN_COUNT):
        if is_valid_location(board, c):
            return False
    return True


# --- Utility Function for AI (Minimax) ---


def get_valid_moves(board):
    """Returns a list of columns where a piece can be dropped."""
    valid_moves = []
    for col in range(COLUMN_COUNT):
        if is_valid_location(board, col):
            valid_moves.append(col)
    return valid_moves


def get_piece_at(board, r, c):
    """Returns the piece at a specific (row, col) or 0 if out of bounds."""
    if 0 <= r < ROW_COUNT and 0 <= c < COLUMN_COUNT:
        return board[r][c]
    return 0
