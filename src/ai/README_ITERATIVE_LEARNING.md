# Iterative Learning for Neural Network AI


- `ai_vs_ai_games.py` - Runs games between AIs and records gameplay
- `retrain_with_new_data.py` - Retrains model with combined datasets
- `optimize_nn_with_iterative_learning.py` - Master script that runs the full pipeline


run this

```bash
cd src/ai
python optimize_nn_with_iterative_learning.py
```

This will:
1. Run 50 games between Heuristic AI and Neural Network AI
2. Save all board states to `data/ai_vs_ai_games_YYYYMMDD_HHMMSS.csv`
3. Retrain the model with original + new data
4. Save the improved model

Test the improved model:

```bash
# In GUI
python src/game/connect_four_gui.py
# Select "Human vs Neural Network AI"

# Or run metrics
python src/ai/run_nn_metrics.py
```

## Data Format

The recorded games use the same format as the original dataset:
- **42 columns**: Board state (flattened 6x7 board)
  - Values: `1.0` (Player 1), `-1.0` (Player 2), `0.0` (Empty)
- **1 column**: Winner
  - Values: `-1.0` (Player 1 wins), `1.0` (Player 2 wins), `0.0` (Draw)


