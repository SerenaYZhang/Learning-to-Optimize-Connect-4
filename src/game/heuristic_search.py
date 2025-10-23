from connect_four_logic import (
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

    # Make sure sign aligns with perspective
    return score if player_num == 1 else -score


def minimax(board, depth, is_maximizing, alph, beta, MAX_DEPTH, human_starts):
    # Check if game terminated
    over_result = check_win(board)
    if over_result == 1:
        return MAX_BASE - 0.25 * MAX_BASE * (depth / MAX_DEPTH)
    if over_result == 2:
        return MIN_BASE + 0.25 * MAX_BASE * (depth / MAX_DEPTH)

    # Heuristic value if at max depth
    if depth >= MAX_DEPTH:
        player_num = 2 if human_starts else 1
        heuristic_value = alt_heuristic_value(board, player_num, False)
        discounted_value = heuristic_value - 0.25 * heuristic_value * (
            depth / MAX_DEPTH
        )
        return discounted_value if player_num == 1 else -discounted_value

    # Game non-terminated
    if is_maximizing:
        best_val = -200
        for j in range(7):
            next_move = get_next_open_row(j, board)
            if next_move is None:
                continue
            child_board = [row[:] for row in board]
            child_board[next_move[0]][next_move[1]] = 1
            child_val = minimax(
                child_board, depth + 1, False, alph, beta, MAX_DEPTH, human_starts
            )
            best_val = max(best_val, child_val)
            alph = max(alph, best_val)
            if beta <= alph:
                break
        return best_val
    else:
        best_val = 200
        for j in range(7):
            next_move = get_next_open_row(j, board)
            if next_move is None:
                continue
            child_board = [row[:] for row in board]
            child_board[next_move[0]][next_move[1]] = 2
            child_val = minimax(
                child_board, depth + 1, True, alph, beta, MAX_DEPTH, human_starts
            )
            best_val = min(best_val, child_val)
            beta = min(beta, best_val)
            if beta <= alph:
                break
        return best_val
