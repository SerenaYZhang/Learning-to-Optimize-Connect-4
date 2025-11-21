"""
Script to evaluate the performance of the trained Connect 4 Neural Network model.

This script evaluates the AI by:
1. Loading the pre-trained Keras model.
2. Running multiple games (NN AI vs Random Player).
3. Tracking performance metrics (win rate, average moves, computation time).
4. Saving results to CSV and JSON files for analysis.
"""

import time
import random
import json
import csv
import sys
import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Attempt to load Keras/TensorFlow components
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    print("Error: TensorFlow/Keras not installed. Please install with 'pip install tensorflow'.")
    sys.exit(1)

# Add parent directory to path for game logic imports (assuming project structure)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import game logic constants and functions
try:
    from game.connect_four_logic import (
        create_board,
        check_win,
        is_board_full,
        get_next_open_row,
        is_valid_location,
        COLUMN_COUNT,
        ROW_COUNT,
    )
    # Define BOARD_SIZE based on the logic file
    BOARD_SIZE = COLUMN_COUNT * ROW_COUNT
except ImportError:
    print("Error: Could not import game logic. Ensure 'game/connect_four_logic.py' exists.")
    sys.exit(1)


# --- Configuration ---
NN_MODEL_PATH = 'connect_four_ai_model.h5'


class GameMetrics:
    """Class to track and store game metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all metrics for a new game."""
        self.total_moves = 0
        self.ai_moves = 0
        self.opponent_moves = 0
        self.ai_move_times = []
        self.game_start_time = None
        self.game_end_time = None
        self.winner = None

    def record_move_time(self, move_time: float):
        """Record the time taken for an AI move."""
        self.ai_move_times.append(move_time)

    def get_average_move_time(self) -> float:
        """Calculate average AI move time."""
        if not self.ai_move_times:
            return 0.0
        return sum(self.ai_move_times) / len(self.ai_move_times)

    def get_total_game_time(self) -> float:
        """Calculate total game time."""
        if self.game_start_time and self.game_end_time:
            return self.game_end_time - self.game_start_time
        return 0.0

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for easy serialization."""
        return {
            'total_moves': self.total_moves,
            'ai_moves': self.ai_moves,
            'opponent_moves': self.opponent_moves,
            'avg_move_time': self.get_average_move_time(),
            'total_game_time': self.get_total_game_time(),
            'winner': self.winner,
        }


def random_move(board) -> Optional[int]:
    """Select a random valid column."""
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    if valid_cols:
        return random.choice(valid_cols)
    return None


def nn_move(board, model: tf.keras.Model, current_player: int) -> Optional[int]:
    """
    Selects the best move using the trained neural network model.

    The model predicts the probability of Player 1 winning [P(-1), P(0), P(1)]
    for a given board state. The AI (Player 1) chooses the move that leads to
    a resulting board state with the highest P(Winner=1) (Index 2).

    The model expects board values: 1.0 (Player 1), -1.0 (Player -1/Opponent), 0.0 (Empty).
    The game uses: 1 (Player 1), 2 (Opponent), 0 (Empty). We must convert 2 -> -1.0.
    """
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    best_col = None
    best_win_prob = -1.0
    
    if not valid_cols:
        return None

    for col in valid_cols:
        # 1. Simulate the move
        # Convert the board copy to a NumPy array to enable .flatten() and other np operations
        temp_board = np.array(board.copy()) 
        
        # FIX: get_next_open_row returns a single row index (int) or None, not a tuple.
        row = get_next_open_row(temp_board, col)
        
        if row is None:
            # This should not happen if is_valid_location passed, but as a safeguard:
            continue

        temp_board[row][col] = current_player

        # 2. Prepare the board for the NN (convert 2 -> -1.0)
        # Flatten the board (42 elements) and change data type to float
        flat_board = temp_board.flatten().astype(float) 
        
        # Map Player 2 (Opponent) back to Player -1 for the model input
        # Note: np.where requires the condition, then the value if true, then the value if false.
        # Since player 1 is already 1.0, and empty is 0.0, we only need to map 2.0 to -1.0.
        nn_input_data = np.where(flat_board == 2.0, -1.0, flat_board).reshape(1, BOARD_SIZE) 
        
        # 3. Predict the outcome: [P(Winner=-1), P(Draw), P(Winner=1)]
        # We use verbose=0 to suppress the Keras output for each prediction
        prediction = model.predict(nn_input_data, verbose=0)[0] 
        
        # AI (Player 1) aims to maximize the probability of winning (Index 2)
        win_prob = prediction[2] 
        
        if win_prob > best_win_prob:
            best_win_prob = win_prob
            best_col = col
        # Tie-breaker: prefer a draw (Index 1) over a loss (Index 0)
        elif win_prob == best_win_prob and prediction[1] > prediction[0]:
            if best_col is None: # Only if a better move hasn't been found yet
                 best_col = col

    return best_col


def play_game(
    model: tf.keras.Model,
    opponent_type: str = "random",
    ai_starts: bool = True,
    verbose: bool = False
) -> GameMetrics:
    """
    Play a single game (NN AI vs Opponent) and collect metrics.

    Args:
        model: The loaded Keras model.
        opponent_type: Type of opponent ('random')
        ai_starts: Whether NN AI goes first
        verbose: Print game progress

    Returns:
        GameMetrics object with collected data
    """
    board = create_board()
    metrics = GameMetrics()
    metrics.game_start_time = time.time()

    # Player 1 is AI, Player 2 is Opponent (Random)
    current_player = 1 if ai_starts else 2
    game_over = False

    if verbose:
        print(f"\n{'='*50}")
        print(f"Starting new game: NN AI vs {opponent_type}")
        print(f"AI starts: {ai_starts}")
        print(f"{'='*50}\n")

    while not game_over:
        metrics.total_moves += 1

        # Determine move based on current player
        if current_player == 1:  # NN AI's turn
            metrics.ai_moves += 1
            move_start = time.time()
            col = nn_move(board, model, current_player)
            move_time = time.time() - move_start
            metrics.record_move_time(move_time)

            if verbose:
                print(f"AI chose column {col} (took {move_time:.3f}s)")

        else:  # Opponent's turn (must be random)
            metrics.opponent_moves += 1
            col = random_move(board)

            if verbose:
                print(f"Opponent chose column {col}")

        # Make the move
        if col is not None and is_valid_location(board, col):
            row = get_next_open_row(board, col)
            if row is not None:
                # The board in play_game should probably also be a NumPy array 
                # if the functions it calls expect it, but we maintain the type
                # from create_board() here for consistency and ensure conversion
                # happens in nn_move.
                board[row][col] = current_player
            else:
                if verbose:
                    print(f"Error: Column {col} is unexpectedly full.")
                break
        else:
            if verbose:
                print("No valid moves available or selected!")
            break

        # Check for win or draw
        winner = check_win(board)
        if winner != 0:
            metrics.winner = "AI" if winner == 1 else "Opponent"
            game_over = True
            if verbose:
                print(f"\n{'*'*50}")
                print(f"{metrics.winner} wins!")
                print(f"{'*'*50}\n")
        elif is_board_full(board):
            metrics.winner = "Draw"
            game_over = True
            if verbose:
                print(f"\n{'*'*50}")
                print("Game ended in a draw!")
                print(f"{'*'*50}\n")

        # Switch player
        current_player = 3 - current_player  # Toggle between 1 and 2

    metrics.game_end_time = time.time()
    return metrics


def run_multiple_games(
    model: tf.keras.Model,
    num_games: int = 100,
    opponent_type: str = "random",
    alternate_starts: bool = True,
    verbose: bool = False
) -> List[GameMetrics]:
    """
    Run multiple games and collect aggregate metrics.
    """
    all_metrics = []

    print(f"\n{'='*60}")
    print(f"Running {num_games} games: NN AI vs {opponent_type}")
    print(f"Alternate starts: {alternate_starts}")
    print(f"{'='*60}\n")

    for i in range(num_games):
        ai_starts = (i % 2 == 0) if alternate_starts else True

        if not verbose and (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games...")

        metrics = play_game(
            model=model,
            opponent_type=opponent_type,
            ai_starts=ai_starts,
            verbose=verbose
        )
        all_metrics.append(metrics)

    return all_metrics


def analyze_metrics(metrics_list: List[GameMetrics]) -> Dict:
    """Analyze aggregate metrics from multiple games."""
    total_games = len(metrics_list)
    ai_wins = sum(1 for m in metrics_list if m.winner == "AI")
    opponent_wins = sum(1 for m in metrics_list if m.winner == "Opponent")
    draws = sum(1 for m in metrics_list if m.winner == "Draw")

    total_moves = [m.total_moves for m in metrics_list]
    ai_move_times = [m.get_average_move_time() for m in metrics_list if m.ai_move_times] # Only average non-zero time
    game_times = [m.get_total_game_time() for m in metrics_list]

    analysis = {
        'total_games': total_games,
        'ai_wins': ai_wins,
        'opponent_wins': opponent_wins,
        'draws': draws,
        'ai_win_rate': (ai_wins / total_games * 100) if total_games > 0 else 0,
        'avg_moves_per_game': sum(total_moves) / len(total_moves) if total_moves else 0,
        'avg_move_time': sum(ai_move_times) / len(ai_move_times) if ai_move_times else 0,
        'avg_game_time': sum(game_times) / len(game_times) if game_times else 0,
        'min_moves': min(total_moves) if total_moves else 0,
        'max_moves': max(total_moves) if total_moves else 0,
    }

    return analysis


def save_metrics_to_csv(
    metrics_list: List[GameMetrics],
    filename: str = None
) -> str:
    """Save detailed metrics to a CSV file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nn_metrics_results_{timestamp}.csv"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'metrics_results_nn')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w', newline='') as csvfile:
        if not metrics_list:
            return filepath

        fieldnames = list(metrics_list[0].to_dict().keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for metrics in metrics_list:
            writer.writerow(metrics.to_dict())

    return filepath


def save_analysis_to_json(analysis: Dict, filename: str = None) -> str:
    """Save analysis results to a JSON file."""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"nn_analysis_{timestamp}.json"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'metrics_results_nn')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as jsonfile:
        json.dump(analysis, jsonfile, indent=4)

    return filepath


def print_analysis(analysis: Dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print("NN AI PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total games played: {analysis['total_games']}")
    print(f"AI wins: {analysis['ai_wins']}")
    print(f"Opponent wins: {analysis['opponent_wins']}")
    print(f"Draws: {analysis['draws']}")
    print(f"AI win rate: {analysis['ai_win_rate']:.2f}%")
    print(f"\nAverage moves per game: {analysis['avg_moves_per_game']:.2f}")
    print(f"Move range: {analysis['min_moves']} - {analysis['max_moves']}")
    print(f"\nAverage NN move time: {analysis['avg_move_time']:.4f}s")
    print(f"Average total game time: {analysis['avg_game_time']:.2f}s")
    print(f"{'='*60}\n")


def main():
    """Main execution function."""
    
    # Check for model file
    if not os.path.exists(NN_MODEL_PATH):
        print(f"Error: Model file not found at '{NN_MODEL_PATH}'.")
        print("Please run 'nn.py' first to train and save the model.")
        return

    # Configuration
    NUM_GAMES = 50
    OPPONENT_TYPE = "random" 
    VERBOSE = False

    try:
        # Load the trained model once
        model = load_model(NN_MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("\n" + "="*60)
    print("CONNECT 4 NEURAL NETWORK METRICS EVALUATION")
    print("="*60)

    # Run games
    metrics_list = run_multiple_games(
        model=model,
        num_games=NUM_GAMES,
        opponent_type=OPPONENT_TYPE,
        alternate_starts=True,
        verbose=VERBOSE
    )

    # Analyze results
    analysis = analyze_metrics(metrics_list)
    print_analysis(analysis)

    # Save results
    csv_path = save_metrics_to_csv(metrics_list)
    json_path = save_analysis_to_json(analysis)

    print(f"Results saved:")
    print(f"  - Detailed metrics: {csv_path}")
    print(f"  - Analysis summary: {json_path}")


if __name__ == "__main__":
    main()