import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Configuration ---
FILE_PATH = '../../data/c4_game_database.csv'
BOARD_SIZE = 42 # 6 rows * 7 columns
RANDOM_SEED = 42

def load_and_prepare_data(filepath):
    """
    Loads the Connect Four game data and prepares it for neural network training.
    Separates features (board state) and the target (winner).
    
    Fix: Use skiprows=1 to ignore the string header row.
    """
    print(f"Loading data from: {filepath}")
    
    # Define dtypes for all 43 columns.
    dtype_map = {i: float for i in range(BOARD_SIZE + 1)} # 42 features + 1 target (43 columns total)
    
    try:
        # Use dtype to force all columns to float, resolving the mixed type issue
        # Use skiprows=1 to skip the header row containing string names
        df = pd.read_csv(filepath, header=None, dtype=dtype_map, skiprows=1)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}. Please ensure the path is correct.")
        return None, None, None

    # Columns 0-41 are features (X), Column 42 (index 42) is the target (Y, winner)
    X = df.iloc[:, :BOARD_SIZE].values
    # Y_raw is now guaranteed to be a uniform numeric type (float)
    Y_raw = df.iloc[:, BOARD_SIZE].values

    # --- Target (Y) Encoding ---
    # The winner column contains -1.0, 0.0, and 1.0.
    
    # 1. Map raw labels (-1.0, 0.0, 1.0) to integers (0, 1, 2)
    label_encoder = LabelEncoder()
    Y_encoded = label_encoder.fit_transform(Y_raw)

    # 2. Convert integer labels to one-hot encoding
    Y_one_hot = to_categorical(Y_encoded)
    
    print(f"Data shape - X: {X.shape}, Y: {Y_one_hot.shape}")
    return X, Y_one_hot, label_encoder.classes_

def build_model(input_dim, output_dim):
    """
    Defines the Keras Sequential neural network model architecture.
    """
    model = Sequential([
        # Input layer: 42 features (board positions)
        Dense(256, activation='relu', input_shape=(input_dim,), name='dense_1'),
        Dropout(0.3),
        
        Dense(128, activation='relu', name='dense_2'),
        Dropout(0.3),
        
        Dense(64, activation='relu', name='dense_3'),
        
        # Output layer: 3 classes (Win -1, Draw, Win 1). Use softmax for multi-class classification.
        Dense(output_dim, activation='softmax', name='output_layer')
    ])
    
    # Compile the model
    # Use 'categorical_crossentropy' for one-hot encoded labels
    model.compile(optimizer='adam', 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])
    
    return model

def main():
    """
    Main function to run the Connect Four AI training process.
    """
    X, Y, classes = load_and_prepare_data(FILE_PATH)
    
    if X is None:
        return

    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=RANDOM_SEED
    )

    print("\n--- Starting Model Training ---")
    model = build_model(input_dim=BOARD_SIZE, output_dim=Y.shape[1])
    model.summary()
    
    # Train the model
    history = model.fit(
        X_train, Y_train, 
        epochs=10, 
        batch_size=256, 
        validation_split=0.1, # Use 10% of training data for validation
        verbose=1
    )
    
    print("\n--- Model Evaluation ---")
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")

    # --- Save the Trained Model ---
    model_filename = 'connect_four_ai_model.h5'
    model.save(model_filename)
    print(f"\nModel successfully trained and saved as '{model_filename}'")
    
    # Note on classes:
    # If the model predicts a vector like [0.1, 0.2, 0.7], the winner prediction 
    # corresponds to the index with the highest probability (index 2 in this case, 
    # which maps back to the winner value 1).
    print(f"Winner mapping for prediction reference: Index 0 -> {classes[0]} (Player -1), Index 1 -> {classes[1]} (Draw), Index 2 -> {classes[2]} (Player 1)")

if __name__ == '__main__':
    # Set a seed for reproducibility
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)
    
    # Suppress TensorFlow logging verbosity
    tf.get_logger().setLevel('ERROR')

    main()