#!/usr/bin/env python3
"""Test script to verify GUI can start and check for errors."""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

print("Testing imports...")

try:
    import tkinter as tk
    print("✓ tkinter imported")
except ImportError as e:
    print(f"✗ tkinter import failed: {e}")
    sys.exit(1)

try:
    from game.connect_four_logic import create_board, ROW_COUNT, COLUMN_COUNT
    print("✓ Game logic imported")
except ImportError as e:
    print(f"✗ Game logic import failed: {e}")
    sys.exit(1)

try:
    from ai.heuristic_search import get_best_move
    print("✓ Heuristic search imported")
except ImportError as e:
    print(f"✗ Heuristic search import failed: {e}")
    sys.exit(1)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    print("✓ TensorFlow imported")
    
    # Check if model file exists
    model_path = os.path.join('src', 'ai', 'connect_four_ai_model.h5')
    if os.path.exists(model_path):
        print(f"✓ Model file found at: {model_path}")
        try:
            model = load_model(model_path)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Model loading failed: {e}")
    else:
        print(f"✗ Model file not found at: {model_path}")
except ImportError:
    print("⚠ TensorFlow not available (this is OK, NN AI just won't work)")

print("\nAll imports successful! Starting GUI...")
print("(Close the GUI window to exit)\n")

# Now try to start the GUI
try:
    from game.connect_four_gui import ConnectFourApp
    
    root = tk.Tk()
    app = ConnectFourApp(root)
    print("GUI started! Window should be visible.")
    root.mainloop()
except Exception as e:
    print(f"✗ Error starting GUI: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

