import math
import random
from connect_four_logic import (
    ROW_COUNT,
    COLUMN_COUNT,
    get_next_open_row,
    is_valid_location,
    check_win,
    is_board_full,
)

MAX_BASE = 80
MIN_BASE = -80
MAX_DEPTH = 4  # Depth limit for minimax

# --- Helper functions ---


def alt_heuristic_value(board, player_num):
    """Weighted heuristic score for Connect 4."""
    opponent = 2 if player_num == 1 else 1
    player_combos = {1: 0, 2: 0, 3: 0}
    blocked_combos = {1: 0, 2: 0, 3: 0}
    total_pieces = sum(cell != 0 for row in board for cell in row)
    if total_pieces == 0:
        total_pieces = 1

    def update_counts(window):
        p_count = window.count(player_num)
        o_count = window.count(opponent)
        if p_count > 0 and o_count > 0:
            if p_count <= 3:
                blocked_combos[p_count] += 1
        elif p_count > 0:
            if p_count <= 3:
                player_combos[p_count] += 1

    # Horizontal
    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            update_counts(board[r][c : c + 4])
    # Vertical
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i][c] for i in range(4)]
            update_counts(window)
    # Positive diagonal
    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            update_counts(window)
    # Negative diagonal
    for r in range(3, ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r - i][c + i] for i in range(4)]
            update_counts(window)

    score = 1.5 * (
        (
            player_combos[1] * 0.5
            + player_combos[2] * 3
            + player_combos[3] * 9
            + blocked_combos[1] * 0.5
            + blocked_combos[2] * 2
            + blocked_combos[3] * 40
        )
        / total_pieces
    )
    return score


def get_next_move(col, board):
    row = get_next_open_row(board, col)
    if row == -1:
        return None
    return (row, col)


def minimax(board, depth, is_maximizing, alpha, beta, ai_player):
    opponent = 1 if ai_player == 2 else 2

    # Check for terminal states
    if check_win(board, ai_player):
        return MAX_BASE - 0.25 * MAX_BASE * (depth / MAX_DEPTH)
    elif check_win(board, opponent):
        return MIN_BASE + 0.25 * MAX_BASE * (depth / MAX_DEPTH)
    elif is_board_full(board) or depth >= MAX_DEPTH:
        heuristic_value = alt_heuristic_value(board, ai_player)
        return heuristic_value if is_maximizing else -heuristic_value

    if is_maximizing:
        best_val = -math.inf
        for col in range(COLUMN_COUNT):
            move = get_next_move(col, board)
            if move is None:
                continue
            new_board = [row[:] for row in board]
            new_board[move[0]][move[1]] = ai_player
            val = minimax(new_board, depth + 1, False, alpha, beta, ai_player)
            best_val = max(best_val, val)
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val
    else:
        best_val = math.inf
        for col in range(COLUMN_COUNT):
            move = get_next_move(col, board)
            if move is None:
                continue
            new_board = [row[:] for row in board]
            new_board[move[0]][move[1]] = opponent
            val = minimax(new_board, depth + 1, True, alpha, beta, ai_player)
            best_val = min(best_val, val)
            beta = min(beta, best_val)
            if beta <= alpha:
                break
        return best_val


def get_best_move(board, ai_player):
    best_score = -math.inf
    best_col = None
    for col in range(COLUMN_COUNT):
        move = get_next_move(col, board)
        if move is None:
            continue
        new_board = [row[:] for row in board]
        new_board[move[0]][move[1]] = ai_player
        score = minimax(new_board, 0, False, -math.inf, math.inf, ai_player)
        if score > best_score:
            best_score = score
            best_col = col
    return best_col
