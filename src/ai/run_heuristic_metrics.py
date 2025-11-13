"""
Script to run heuristic search AI multiple times and record performance metrics.

This script evaluates the Connect 4 AI using minimax with alpha-beta pruning by:
1. Running multiple games (AI vs AI, AI vs Random, etc.)
2. Tracking performance metrics (win rate, average moves, computation time, pruning efficiency)
3. Saving results to a CSV file for analysis

Author: Nicole Sin (ns753)
"""

import time
import random
import json
import csv
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.connect_four_logic import (
    create_board,
    check_win,
    is_board_full,
    get_next_open_row,
    is_valid_location,
    COLUMN_COUNT,
)
from ai.heuristic_search import get_best_move, minimax, MAX_DEPTH


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
        self.nodes_explored = 0
        self.nodes_pruned = 0
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

    def get_pruning_efficiency(self) -> float:
        """Calculate what percentage of nodes were pruned."""
        total = self.nodes_explored + self.nodes_pruned
        if total == 0:
            return 0.0
        return (self.nodes_pruned / total) * 100

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary for easy serialization."""
        return {
            'total_moves': self.total_moves,
            'ai_moves': self.ai_moves,
            'opponent_moves': self.opponent_moves,
            'avg_move_time': self.get_average_move_time(),
            'total_game_time': self.get_total_game_time(),
            'nodes_explored': self.nodes_explored,
            'nodes_pruned': self.nodes_pruned,
            'pruning_efficiency': self.get_pruning_efficiency(),
            'winner': self.winner,
        }


def random_move(board) -> Optional[int]:
    """Select a random valid column."""
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    if valid_cols:
        return random.choice(valid_cols)
    return None


def play_game(
    opponent_type: str = "random",
    max_depth: int = MAX_DEPTH,
    ai_starts: bool = True,
    verbose: bool = False
) -> GameMetrics:
    """
    Play a single game and collect metrics.

    Args:
        opponent_type: Type of opponent ('random', 'ai', 'minimax')
        max_depth: Maximum search depth for AI
        ai_starts: Whether AI goes first
        verbose: Print game progress

    Returns:
        GameMetrics object with collected data
    """
    board = create_board()
    metrics = GameMetrics()
    metrics.game_start_time = time.time()

    current_player = 1 if ai_starts else 2
    game_over = False

    if verbose:
        print(f"\n{'='*50}")
        print(f"Starting new game: AI vs {opponent_type}")
        print(f"AI starts: {ai_starts}, Max depth: {max_depth}")
        print(f"{'='*50}\n")

    while not game_over:
        metrics.total_moves += 1

        # Determine move based on current player
        if current_player == 1:  # AI's turn
            metrics.ai_moves += 1
            move_start = time.time()
            col = get_best_move(board, max_depth=max_depth, human_starts=not ai_starts)
            move_time = time.time() - move_start
            metrics.record_move_time(move_time)

            if verbose:
                print(f"AI chose column {col} (took {move_time:.3f}s)")

        else:  # Opponent's turn
            metrics.opponent_moves += 1
            if opponent_type == "random":
                col = random_move(board)
            elif opponent_type == "ai":
                # Second AI with same settings
                col = get_best_move(board, max_depth=max_depth, human_starts=ai_starts)
            else:
                col = random_move(board)  # Default to random

            if verbose:
                print(f"Opponent chose column {col}")

        # Make the move
        if col is not None:
            move = get_next_open_row(col, board)
            if move:
                board[move[0]][move[1]] = current_player
            else:
                if verbose:
                    print(f"Invalid move attempted: column {col}")
                break
        else:
            if verbose:
                print("No valid moves available!")
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
    num_games: int = 100,
    opponent_type: str = "random",
    max_depth: int = MAX_DEPTH,
    alternate_starts: bool = True,
    verbose: bool = False
) -> List[GameMetrics]:
    """
    Run multiple games and collect aggregate metrics.

    Args:
        num_games: Number of games to play
        opponent_type: Type of opponent ('random', 'ai')
        max_depth: Maximum search depth for AI
        alternate_starts: Whether to alternate who starts
        verbose: Print detailed progress

    Returns:
        List of GameMetrics objects
    """
    all_metrics = []

    print(f"\n{'='*60}")
    print(f"Running {num_games} games: AI vs {opponent_type}")
    print(f"Max depth: {max_depth}, Alternate starts: {alternate_starts}")
    print(f"{'='*60}\n")

    for i in range(num_games):
        ai_starts = (i % 2 == 0) if alternate_starts else True

        if not verbose and (i + 1) % 10 == 0:
            print(f"Completed {i + 1}/{num_games} games...")

        metrics = play_game(
            opponent_type=opponent_type,
            max_depth=max_depth,
            ai_starts=ai_starts,
            verbose=verbose
        )
        all_metrics.append(metrics)

    return all_metrics


def analyze_metrics(metrics_list: List[GameMetrics]) -> Dict:
    """
    Analyze aggregate metrics from multiple games.

    Args:
        metrics_list: List of GameMetrics objects

    Returns:
        Dictionary with aggregate statistics
    """
    total_games = len(metrics_list)
    ai_wins = sum(1 for m in metrics_list if m.winner == "AI")
    opponent_wins = sum(1 for m in metrics_list if m.winner == "Opponent")
    draws = sum(1 for m in metrics_list if m.winner == "Draw")

    total_moves = [m.total_moves for m in metrics_list]
    avg_move_times = [m.get_average_move_time() for m in metrics_list]
    game_times = [m.get_total_game_time() for m in metrics_list]

    analysis = {
        'total_games': total_games,
        'ai_wins': ai_wins,
        'opponent_wins': opponent_wins,
        'draws': draws,
        'ai_win_rate': (ai_wins / total_games * 100) if total_games > 0 else 0,
        'avg_moves_per_game': sum(total_moves) / len(total_moves) if total_moves else 0,
        'avg_move_time': sum(avg_move_times) / len(avg_move_times) if avg_move_times else 0,
        'avg_game_time': sum(game_times) / len(game_times) if game_times else 0,
        'min_moves': min(total_moves) if total_moves else 0,
        'max_moves': max(total_moves) if total_moves else 0,
    }

    return analysis


def save_metrics_to_csv(
    metrics_list: List[GameMetrics],
    filename: str = None
) -> str:
    """
    Save metrics to a CSV file.

    Args:
        metrics_list: List of GameMetrics objects
        filename: Output filename (auto-generated if None)

    Returns:
        Path to the saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_results_{timestamp}.csv"

    # Ensure we're saving to the correct directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'metrics_results')
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
    """
    Save analysis results to a JSON file.

    Args:
        analysis: Dictionary with analysis results
        filename: Output filename (auto-generated if None)

    Returns:
        Path to the saved file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_{timestamp}.json"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'metrics_results')
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)

    with open(filepath, 'w') as jsonfile:
        json.dump(analysis, jsonfile, indent=4)

    return filepath


def print_analysis(analysis: Dict):
    """Pretty print analysis results."""
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total games played: {analysis['total_games']}")
    print(f"AI wins: {analysis['ai_wins']}")
    print(f"Opponent wins: {analysis['opponent_wins']}")
    print(f"Draws: {analysis['draws']}")
    print(f"AI win rate: {analysis['ai_win_rate']:.2f}%")
    print(f"\nAverage moves per game: {analysis['avg_moves_per_game']:.2f}")
    print(f"Move range: {analysis['min_moves']} - {analysis['max_moves']}")
    print(f"\nAverage AI move time: {analysis['avg_move_time']:.4f}s")
    print(f"Average total game time: {analysis['avg_game_time']:.2f}s")
    print(f"{'='*60}\n")


def main():
    """Main execution function."""
    # Configuration
    NUM_GAMES = 50
    OPPONENT_TYPE = "random"  # Options: 'random', 'ai'
    MAX_DEPTH_TEST = 5
    VERBOSE = False

    print("\n" + "="*60)
    print("CONNECT 4 HEURISTIC SEARCH METRICS EVALUATION")
    print("="*60)

    # Run games
    metrics_list = run_multiple_games(
        num_games=NUM_GAMES,
        opponent_type=OPPONENT_TYPE,
        max_depth=MAX_DEPTH_TEST,
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

    # Additional test with different depth
    if True:  # Set to True to test multiple depths
        print("\n" + "="*60)
        print("TESTING DIFFERENT SEARCH DEPTHS")
        print("="*60 + "\n")

        depths_to_test = [3, 5, 7]
        depth_results = {}

        for depth in depths_to_test:
            print(f"\nTesting depth {depth}...")
            metrics = run_multiple_games(
                num_games=20,
                opponent_type="random",
                max_depth=depth,
                alternate_starts=True,
                verbose=False
            )
            analysis = analyze_metrics(metrics)
            depth_results[depth] = analysis

            print(f"Depth {depth}: Win rate = {analysis['ai_win_rate']:.2f}%, "
                  f"Avg move time = {analysis['avg_move_time']:.4f}s")

        # Save depth comparison
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        depth_file = f"depth_comparison_{timestamp}.json"
        depth_path = save_analysis_to_json(depth_results, depth_file)
        print(f"\nDepth comparison saved to: {depth_path}")


if __name__ == "__main__":
    main()
