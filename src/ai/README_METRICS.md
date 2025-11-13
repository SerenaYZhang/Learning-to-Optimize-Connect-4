# Connect 4 Heuristic Search Metrics Script

## Overview
This script evaluates the performance of the Connect 4 AI using minimax with alpha-beta pruning. It runs multiple games and collects comprehensive metrics to analyze the AI's performance.

**Author:** Nicole Sin (ns753)

## Features

### Alpha-Beta Pruning
The minimax implementation includes **alpha-beta pruning** for improved efficiency:
- **Alpha**: Best value the maximizer can guarantee at current node
- **Beta**: Best value the minimizer can guarantee at current node
- **Pruning**: When `beta <= alpha`, the branch is cut off since it won't affect the final decision
- This dramatically reduces the number of nodes explored in the game tree

### Metrics Tracked
For each game, the script tracks:
- Total moves in the game
- Number of AI moves vs opponent moves
- Average time per AI move
- Total game time
- Winner (AI, Opponent, or Draw)
- Win rate statistics
- Move count ranges (min/max)

### Aggregate Analysis
The script provides:
- Win rate percentage
- Average moves per game
- Average computation time per move
- Total game duration statistics
- Comparison across different search depths

## Usage

### Basic Usage
Run the script with default settings (50 games, depth 5, vs random opponent):
```bash
cd /Users/nicolesin/Documents/dev_env/Learning-to-Optimize-Connect-4
python src/ai/run_heuristic_metrics.py
```

### Configuration
Edit the `main()` function in the script to customize:

```python
# Number of games to run
NUM_GAMES = 50

# Opponent type: 'random' or 'ai'
OPPONENT_TYPE = "random"

# Maximum search depth for minimax
MAX_DEPTH_TEST = 5

# Print detailed game progress
VERBOSE = False
```

### Opponent Types
- **'random'**: Random move selection (good for baseline testing)
- **'ai'**: Another AI with same settings (AI vs AI)

### Search Depth Testing
The script automatically tests multiple search depths (3, 5, 7) and compares:
- Win rates at each depth
- Average computation time per move
- Trade-off between performance and speed

## Output Files

All results are saved to `src/ai/metrics_results/` directory:

### 1. CSV File: `metrics_results_YYYYMMDD_HHMMSS.csv`
Detailed per-game metrics in CSV format:
- Columns: total_moves, ai_moves, opponent_moves, avg_move_time, total_game_time, winner
- One row per game
- Easy to import into Excel, pandas, or other analysis tools

### 2. JSON File: `analysis_YYYYMMDD_HHMMSS.json`
Aggregate statistics in JSON format:
```json
{
    "total_games": 50,
    "ai_wins": 50,
    "opponent_wins": 0,
    "draws": 0,
    "ai_win_rate": 100.0,
    "avg_moves_per_game": 10.58,
    "avg_move_time": 0.0871,
    "avg_game_time": 0.483,
    "min_moves": 7,
    "max_moves": 25
}
```

### 3. Depth Comparison: `depth_comparison_YYYYMMDD_HHMMSS.json`
Comparison of performance across different search depths

## Sample Results

### Performance vs Random Opponent (Depth 5, 50 games)
```
Total games played: 50
AI wins: 50
Opponent wins: 0
Draws: 0
AI win rate: 100.00%

Average moves per game: 10.58
Move range: 7 - 25

Average AI move time: 0.0871s
Average total game time: 0.48s
```

### Depth Comparison (20 games each)
| Depth | Win Rate | Avg Move Time | Analysis |
|-------|----------|---------------|----------|
| 3     | 95.00%   | 0.0061s      | Fast but less reliable |
| 5     | 100.00%  | 0.0993s      | Good balance (recommended) |
| 7     | 100.00%  | 1.7008s      | Most thorough but slow |

## Key Findings

### Alpha-Beta Pruning Effectiveness
The alpha-beta pruning implementation significantly reduces the search space:
- Without pruning: O(b^d) nodes explored (b=7 branches, d=depth)
- With pruning: O(b^(d/2)) nodes in best case
- For depth 5: ~16,807 nodes → ~343 nodes (theoretical best case)

### Optimal Configuration
Based on testing:
- **Recommended depth**: 5
  - 100% win rate vs random
  - ~0.09s per move (very responsive)
  - Good balance of performance and speed

- **For competitive play**: Depth 7
  - Maximum performance
  - Longer computation time acceptable for tournament play

- **For real-time play**: Depth 3
  - Near-instant moves
  - Still 95% win rate

## Integration with Your Project

### Using the AI in Your Game
```python
from ai.heuristic_search import get_best_move
from game.connect_four_logic import create_board

# Initialize game
board = create_board()

# Get AI move
ai_column = get_best_move(board, max_depth=5, human_starts=True)

# Make the move
# ... (use your drop_piece logic)
```

### Running Custom Tests
```python
from ai.run_heuristic_metrics import run_multiple_games, analyze_metrics

# Run 100 games vs AI opponent
metrics = run_multiple_games(
    num_games=100,
    opponent_type="ai",
    max_depth=5,
    alternate_starts=True,
    verbose=False
)

# Analyze results
analysis = analyze_metrics(metrics)
print(f"Win rate: {analysis['ai_win_rate']:.2f}%")
```

## Future Enhancements

Potential improvements to track:
1. **Node counting**: Instrument minimax to count explored vs pruned nodes
2. **Opening book**: Track which opening moves lead to fastest wins
3. **Position evaluation**: Analyze heuristic values at critical game states
4. **Neural network comparison**: Compare against trained NN model

## Dependencies

- Python 3.6+
- Standard library modules: time, random, json, csv, datetime, sys, os
- Local modules: game.connect_four_logic, ai.heuristic_search

## Troubleshooting

### Import Errors
If you get `ModuleNotFoundError`, ensure you're running from the project root:
```bash
cd /Users/nicolesin/Documents/dev_env/Learning-to-Optimize-Connect-4
python src/ai/run_heuristic_metrics.py
```

### Slow Performance
If games are taking too long:
- Reduce `MAX_DEPTH` (try 3 or 4)
- Reduce `NUM_GAMES`
- Use 'random' opponent instead of 'ai'

### Memory Issues
For very large test runs (1000+ games):
- Process results in batches
- Clear metrics list periodically
- Use generator pattern instead of storing all results

## Contact

For questions or improvements, contact:
- Nicole Sin (ns753)
- Team: Ryan Park (ryp3), Serena Zhang (syz8)
