# run games between heuristic ai and nn ai to get training data

import sys
import os
import numpy as np
import pandas as pd
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import load_model

from game.connect_four_logic import (
    create_board,
    check_win,
    is_board_full,
    get_next_open_row,
    is_valid_location,
    drop_piece,
    COLUMN_COUNT,
    ROW_COUNT,
)
from ai.heuristic_search import get_best_move

BOARD_SIZE = ROW_COUNT * COLUMN_COUNT
NN_MODEL_PATH = 'connect_four_ai_model.h5'


def nn_move(board, model, current_player):
    # get best move from nn model
    valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(board, col)]
    if not valid_cols:
        return None
    
    best_col = None
    best_prob = -1
    
    for col in valid_cols:
        temp_board = np.array([row[:] for row in board])
        row = get_next_open_row(temp_board, col)
        if row == -1:
            continue
        
        temp_board[row][col] = current_player
        
        # convert to model format (1->1.0, 2->-1.0, 0->0.0)
        flat = temp_board.flatten().astype(float)
        nn_input = np.where(flat == 2.0, -1.0, flat).reshape(1, BOARD_SIZE)
        
        pred = model.predict(nn_input, verbose=0)[0]
        
        if current_player == 1:
            prob = pred[2]  # player 1 wins
        else:
            prob = pred[0]  # player 2 wins
        
        if prob > best_prob:
            best_prob = prob
            best_col = col
    
    return best_col


def play_ai_vs_ai_game(model, heuristic_starts=True):
    # play one game: heuristic ai (player 1) vs nn ai (player 2)
    board = create_board()
    states = []
    current_player = 1 if heuristic_starts else 2
    
    while True:
        # save board state
        board_copy = np.array([row[:] for row in board]).astype(float)
        board_copy = np.where(board_copy == 2.0, -1.0, board_copy)
        states.append(board_copy.flatten())
        
        # make move
        if current_player == 1:
            # heuristic ai at depth 5
            col = get_best_move(board, ai_player=1, max_depth=5)
        else:
            col = nn_move(board, model, current_player)
        
        if col is None:
            valid = [c for c in range(COLUMN_COUNT) if is_valid_location(board, c)]
            if valid:
                col = valid[0]
            else:
                break
        
        row = get_next_open_row(board, col)
        if row == -1:
            break
        
        drop_piece(board, row, col, current_player)
        
        # check win
        if check_win(board, current_player):
            winner = -1.0 if current_player == 1 else 1.0
            break
        elif is_board_full(board):
            winner = 0.0
            break
        
        current_player = 3 - current_player
    
    # save final state
    board_copy = np.array([row[:] for row in board]).astype(float)
    board_copy = np.where(board_copy == 2.0, -1.0, board_copy)
    states.append(board_copy.flatten())
    
    return states, winner


def run_games(model, num_games=50):
    # run multiple games and collect data
    all_data = []
    h_wins = 0
    nn_wins = 0
    draws = 0
    
    print(f"Running {num_games} games...")
    
    for i in range(num_games):
        if (i + 1) % 10 == 0:
            print(f"Game {i+1}/{num_games}")
        
        # alternate who starts
        h_starts = (i % 2 == 0)
        states, winner = play_ai_vs_ai_game(model, h_starts)
        
        # add all states with winner
        for state in states:
            all_data.append(list(state) + [winner])
        
        if winner == -1.0:
            h_wins += 1
        elif winner == 1.0:
            nn_wins += 1
        else:
            draws += 1
    
    print(f"\nResults:")
    print(f"Heuristic wins: {h_wins}")
    print(f"NN wins: {nn_wins}")
    print(f"Draws: {draws}")
    
    # make dataframe
    cols = [f'f{i}' for i in range(BOARD_SIZE)] + ['winner']
    df = pd.DataFrame(all_data, columns=cols)
    return df


def save_data(df):
    # save to csv
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"ai_vs_ai_games_{timestamp}.csv"
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    filepath = os.path.join(data_dir, filename)
    
    df.to_csv(filepath, index=False, header=False)
    print(f"Saved to {filepath}")
    return filepath


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, NN_MODEL_PATH)
    
    if not os.path.exists(model_path):
        print("Model file not found. Run nn.py first.")
        return
    
    model = load_model(model_path)
    print("Model loaded")
    
    num_games = 50
    data = run_games(model, num_games)
    save_data(data)
    
    print(f"Done. Recorded {len(data)} board states from {num_games} games")


if __name__ == "__main__":
    main()

