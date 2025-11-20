import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect_four_logic import (
    ROW_COUNT,
    COLUMN_COUNT,
    check_win,
    drop_piece,
    is_valid_location,
    get_next_open_row,
    is_board_full,
)

# Source: "https://www.deepexploration.org/blog/minimax-algorithm-for-connect-4"
MAX_BASE = 80
MIN_BASE = -80
MAX_DEPTH = 5  # default depth (can be overridden)
# human_starts variable isn't needed for correct perspective if we pass ai_player explicitly


def alt_heuristic_value(board, player_num, debug=False):
    """
    Weighted heuristic for Connect Four.
    Returns a positive score expressing how favorable the board is for player_num.
    """
    opponent = 2 if player_num == 1 else 1
    player_combos = {1: 0, 2: 0, 3: 0}
    blocked_combos = {1: 0, 2: 0, 3: 0}
    total_pieces = sum(cell != 0 for row in board for cell in row)
    if total_pieces == 0:
        total_pieces = 1  # avoid division by zero early in game

    def update_counts(window):
        p_count = window.count(player_num)
        o_count = window.count(opponent)
        if p_count > 0 and o_count > 0:
            # blocked combo (both players present)
            if p_count <= 3 and o_count <= 3:
                blocked_combos[p_count] += 1
        elif p_count > 0 and o_count == 0:
            # potential combo for player
            if p_count <= 3:
                player_combos[p_count] += 1

    for r in range(ROW_COUNT):
        for c in range(COLUMN_COUNT - 3):
            update_counts(board[r][c : c + 4])

    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT - 3):
            window = [board[r + i][c] for i in range(4)]
            update_counts(window)

    for r in range(ROW_COUNT - 3):
        for c in range(COLUMN_COUNT - 3):
            window = [board[r + i][c + i] for i in range(4)]
            update_counts(window)

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

    # Important: return the score from the requested player's perspective (positive)
    return score


def get_best_move(board, ai_player=2, max_depth=MAX_DEPTH):
    """
    Returns the best column for ai_player (1 or 2).
    """
    best_val = float("-inf")
    best_col = None
    alpha = float("-inf")
    beta = float("inf")

    # Try all columns
    for col in range(COLUMN_COUNT):
        row = get_next_open_row(board, col)  # correct argument order
        if row == -1:
            continue

        # Create child board and make move for ai_player
        child_board = [r[:] for r in board]
        child_board[row][col] = ai_player

        # Evaluate this move: next is minimizing (opponent)
        move_val = minimax(child_board, 1, False, alpha, beta, max_depth, ai_player)

        if move_val > best_val:
            best_val = move_val
            best_col = col

        # Update alpha for ordering (not necessary but consistent)
        alpha = max(alpha, best_val)

    return best_col


def minimax(board, depth, is_maximizing, alpha, beta, max_depth, ai_player):
    """
    Minimax algorithm with alpha-beta pruning.
    ai_player: 1 or 2 (which side the AI plays)
    """
    opponent = 1 if ai_player == 2 else 2

    # Terminal checks: check explicit winners for each side
    if check_win(board, ai_player):
        return MAX_BASE - 0.25 * MAX_BASE * (depth / max_depth)
    if check_win(board, opponent):
        return MIN_BASE + 0.25 * MAX_BASE * (depth / max_depth)

    if is_board_full(board):
        return 0

    # Heuristic at max depth
    if depth >= max_depth:
        heuristic_value = alt_heuristic_value(board, ai_player, False)
        discounted_value = heuristic_value - 0.25 * heuristic_value * (
            depth / max_depth
        )
        return discounted_value

    # Explore children
    if is_maximizing:
        best_val = float("-inf")
        for j in range(COLUMN_COUNT):
            row = get_next_open_row(board, j)
            if row == -1:
                continue

            child_board = [r[:] for r in board]
            child_board[row][j] = ai_player

            child_val = minimax(
                child_board, depth + 1, False, alpha, beta, max_depth, ai_player
            )
            best_val = max(best_val, child_val)

            alpha = max(alpha, best_val)
            if beta <= alpha:
                break  # beta cutoff

        return best_val
    else:
        best_val = float("inf")
        for j in range(COLUMN_COUNT):
            row = get_next_open_row(board, j)
            if row == -1:
                continue

            child_board = [r[:] for r in board]
            child_board[row][j] = opponent

            child_val = minimax(
                child_board, depth + 1, True, alpha, beta, max_depth, ai_player
            )
            best_val = min(best_val, child_val)

            beta = min(beta, best_val)
            if beta <= alpha:
                break  # alpha cutoff

        return best_val
