# retrain nn with original data + new ai vs ai games

import sys
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
import shutil

ORIGINAL_DATA = '../../data/c4_game_database.csv'
BOARD_SIZE = 42
RANDOM_SEED = 42
MODEL_PATH = 'connect_four_ai_model.h5'

def find_ai_files():
    # find all ai vs ai csv files
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, '..', 'data')
    if not os.path.exists(data_dir):
        return []
    files = [f for f in os.listdir(data_dir) if f.startswith('ai_vs_ai_games_') and f.endswith('.csv')]
    return [os.path.join(data_dir, f) for f in files]


def load_original(filepath):
    # load original kaggle data
    print(f"Loading original data...")
    try:
        df = pd.read_csv(filepath, header=None, dtype={i: float for i in range(BOARD_SIZE + 1)}, skiprows=1)
        print(f"Loaded {len(df)} samples")
        return df
    except:
        print("Original data not found")
        return pd.DataFrame()


def load_ai_data(filepaths):
    # load ai vs ai game data
    if not filepaths:
        print("No AI vs AI files found")
        return pd.DataFrame()
    
    all_data = []
    for f in filepaths:
        try:
            df = pd.read_csv(f, header=None, dtype={i: float for i in range(BOARD_SIZE + 1)})
            all_data.append(df)
            print(f"Loaded {len(df)} from {os.path.basename(f)}")
        except:
            pass
    
    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        print(f"Total AI data: {len(combined)}")
        return combined
    return pd.DataFrame()


def combine_data(orig_df, new_df):
    # combine datasets
    if new_df.empty:
        return orig_df
    
    combined = pd.concat([orig_df, new_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"Combined: {len(combined)} total")
    print(f"  Original: {len(orig_df)}")
    print(f"  New: {len(new_df)}")
    
    return combined


def prepare_data(df):
    # prepare for training
    if df.empty:
        return None, None, None
    
    X = df.iloc[:, :BOARD_SIZE].values
    Y_raw = df.iloc[:, BOARD_SIZE].values
    
    encoder = LabelEncoder()
    Y_encoded = encoder.fit_transform(Y_raw)
    Y_onehot = to_categorical(Y_encoded)
    
    return X, Y_onehot, encoder.classes_


def build_model(input_dim, output_dim):
    # same model as nn.py
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_dim,)),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(output_dim, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def main():
    print("Retraining model with new data...")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    orig_path = os.path.join(script_dir, ORIGINAL_DATA)
    
    orig_df = load_original(orig_path)
    ai_files = find_ai_files()
    new_df = load_ai_data(ai_files)
    
    combined = combine_data(orig_df, new_df)
    
    if combined.empty:
        print("No data!")
        return
    
    X, Y, classes = prepare_data(combined)
    if X is None:
        return
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_SEED)
    
    print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")
    
    model = build_model(BOARD_SIZE, Y.shape[1])
    model.summary()
    
    print("\nTraining...")
    model.fit(X_train, Y_train, epochs=10, batch_size=256, validation_split=0.1, verbose=1)
    
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print(f"\nTest loss: {loss:.4f}, accuracy: {acc:.4f}")
    
    # backup old model
    model_path = os.path.join(script_dir, MODEL_PATH)
    if os.path.exists(model_path):
        backup = model_path.replace('.h5', f'_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.h5')
        shutil.copy(model_path, backup)
        print(f"Backed up to {backup}")
    
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print("Done!")


if __name__ == '__main__':
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    tf.get_logger().setLevel('ERROR')
    
    main()

