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
MAX_DEPTH = 5  # Example depth limit — adjust as needed
human_starts = True  # Example flag


def alt_heuristic_value(board, player_num, debug=False):
    """
    Weighted heuristic for Connect Four.

    Returns a normalized score based on:
    - number and strength of potential combos for the current player
    - number of blocked combos (windows containing both players)
    Formula (adapted from your article):

    score = 1.5 * (
        playerCombos[1]*0.5 +
        playerCombos[2]*3 +
        playerCombos[3]*9 +
        blockedCombos[1]*0.5 +
        blockedCombos[2]*2 +
        blockedCombos[3]*40
    ) / total_pieces
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
        empty = window.count(0)
        if p_count > 0 and o_count > 0:
            # blocked combo (both players present)
            if p_count <= 3 and o_count <= 3:
                blocked_combos[p_count] += 1
        elif p_count > 0 and o_count == 0:
            # potential combo for player
            if p_count <= 3:
                player_combos[p_count] += 1

    # Generate all 4-cell windows
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

    return score if player_num == 1 else -score


def get_best_move(board, max_depth=MAX_DEPTH, human_starts=True):
    """
    Determines the best move for the AI (player 1) using minimax with alpha-beta pruning.

    Args:
        board: Current game board state
        max_depth: Maximum search depth for minimax
        human_starts: Whether human (player 2) starts the game

    Returns:
        Column number (0-6) for the best move, or None if no valid moves
    """
    best_val = float('-inf')
    best_col = None
    alpha = float('-inf')
    beta = float('inf')

    for col in range(COLUMN_COUNT):
        next_move = get_next_open_row(col, board)
        if next_move is None:
            continue

        # Create child board and make move
        child_board = [row[:] for row in board]
        child_board[next_move[0]][next_move[1]] = 1

        # Evaluate this move
        move_val = minimax(
            child_board, 1, False, alpha, beta, max_depth, human_starts
        )

        if move_val > best_val:
            best_val = move_val
            best_col = col

        # Update alpha
        alpha = max(alpha, best_val)

    return best_col


def minimax(board, depth, is_maximizing, alpha, beta, max_depth, human_starts):
    """
    Minimax algorithm with alpha-beta pruning for Connect 4.

    Args:
        board: Current game board state
        depth: Current depth in the search tree
        is_maximizing: True if maximizing player's turn, False otherwise
        alpha: Best value the maximizer can guarantee (alpha cutoff)
        beta: Best value the minimizer can guarantee (beta cutoff)
        max_depth: Maximum search depth
        human_starts: Whether human (player 2) starts the game

    Returns:
        Best evaluation score for the current position

    Alpha-Beta Pruning:
        - Alpha: The best value that the maximizer currently can guarantee
        - Beta: The best value that the minimizer currently can guarantee
        - Pruning occurs when beta <= alpha (the branch cannot influence the final decision)
    """
    # Check if game terminated
    over_result = check_win(board)
    if over_result == 1:
        #AI wins return high score, decreasing with depth (prefer quicker wins)
        return MAX_BASE - 0.25 * MAX_BASE * (depth / max_depth)
    if over_result == 2:
        # uman wins return low score, increasing with depth (delay losses)
        return MIN_BASE + 0.25 * MAX_BASE * (depth / max_depth)

    #Check for draw
    if is_board_full(board):
        return 0

    #Heuristic value if at max depth
    if depth >= max_depth:
        player_num = 2 if human_starts else 1
        heuristic_value = alt_heuristic_value(board, player_num, False)
        discounted_value = heuristic_value - 0.25 * heuristic_value * (
            depth / max_depth
        )
        return discounted_value if player_num == 1 else -discounted_value

    # Game non-terminated- explore child nodes
    if is_maximizing:
        #Maximizing player
        best_val = float('-inf')
        for j in range(COLUMN_COUNT):
            next_move = get_next_open_row(j, board)
            if next_move is None:
                continue

            #Create child board and make move
            child_board = [row[:] for row in board]
            child_board[next_move[0]][next_move[1]] = 1

            #Recursive call
            child_val = minimax(
                child_board, depth + 1, False, alpha, beta, max_depth, human_starts
            )
            best_val = max(best_val, child_val)

            # Alpha-beta pruning
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break  # Beta cutoff - minimizer won't allow this branch

        return best_val
    else:
        #Minimizing player
        best_val = float('inf')
        for j in range(COLUMN_COUNT):
            next_move = get_next_open_row(j, board)
            if next_move is None:
                continue

            #Create child board and make move
            child_board = [row[:] for row in board]
            child_board[next_move[0]][next_move[1]] = 2

            #Recursive call
            child_val = minimax(
                child_board, depth + 1, True, alpha, beta, max_depth, human_starts
            )
            best_val = min(best_val, child_val)

            # Alpha-beta pruning
            beta = min(beta, best_val)
            if beta <= alpha:
                break  

        return best_val
