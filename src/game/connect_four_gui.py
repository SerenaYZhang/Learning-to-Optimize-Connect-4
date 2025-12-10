import tkinter as tk
from tkinter import messagebox
import math
import random
from connect_four_logic import (
    ROW_COUNT,
    COLUMN_COUNT,
    create_board,
    drop_piece,
    is_valid_location,
    get_next_open_row,
    check_win,
    is_board_full,
)
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ai.heuristic_search import get_best_move

# Try to import TensorFlow for neural network AI
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available. Neural Network AI will not work.")


# --- GUI Constants ---
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 5)
WIDTH = COLUMN_COUNT * SQUARE_SIZE
HEIGHT = (ROW_COUNT + 1) * SQUARE_SIZE
PLAYER1_COLOR = "red"
PLAYER2_COLOR = "yellow"
BOARD_COLOR = "#3498db"
EMPTY_COLOR = "white"
BACKGROUND_COLOR = "#2c3e50"


def draw_board(canvas, board):
    """Draws the Connect 4 board and all current pieces on the canvas."""
    canvas.create_rectangle(
        0, SQUARE_SIZE, WIDTH, HEIGHT, fill=BOARD_COLOR, outline=BOARD_COLOR
    )
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            x = c * SQUARE_SIZE + SQUARE_SIZE / 2
            y = HEIGHT - (r * SQUARE_SIZE + SQUARE_SIZE / 2)
            x1, y1, x2, y2 = x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS
            fill_color = (
                PLAYER1_COLOR
                if board[r][c] == 1
                else PLAYER2_COLOR if board[r][c] == 2 else EMPTY_COLOR
            )
            canvas.create_oval(x1, y1, x2, y2, fill=fill_color, outline="#333", width=2)
    canvas.update()


class ConnectFourApp:
    def __init__(self, master):
        self.master = master
        master.title("Connect 4")
        master.configure(bg=BACKGROUND_COLOR)

        # --- Initialize game state ---
        self.board = create_board()
        self.turn = 1
        self.game_over = False
        self.game_mode = None  # Will be chosen via prompt
        
        # --- Load neural network model if available ---
        self.nn_model = None
        if TENSORFLOW_AVAILABLE:
            try:
                # Path to the model file (relative to this file)
                script_dir = os.path.dirname(os.path.abspath(__file__))
                model_path = os.path.join(script_dir, '..', 'ai', 'connect_four_ai_model.h5')
                model_path = os.path.normpath(model_path)
                
                if os.path.exists(model_path):
                    self.nn_model = load_model(model_path)
                    print(f"Neural network model loaded successfully from {model_path}")
                else:
                    print(f"Warning: Neural network model not found at {model_path}")
            except Exception as e:
                print(f"Warning: Could not load neural network model: {e}")
                self.nn_model = None

        # --- Canvas setup ---
        self.canvas = tk.Canvas(
            master,
            width=WIDTH,
            height=HEIGHT,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(pady=20, padx=20)
        draw_board(self.canvas, self.board)

        # --- Status Label ---
        self.status_label = tk.Label(
            master,
            text="Select a game mode to start",
            font=("Inter", 16, "bold"),
            fg="white",
            bg=BACKGROUND_COLOR,
        )
        self.status_label.pack(pady=10)

        # --- Reset Button ---
        self.reset_button = tk.Button(
            master,
            text="New Game / Change Mode",
            command=self.reset_game,
            font=("Inter", 12),
            bg="#e74c3c",
            fg="black",
            activebackground="#c0392b",
        )
        self.reset_button.pack(pady=5)

        # --- Bindings ---
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_motion)
        self.indicator_circle = None

        # --- Show game mode selection at startup ---
        self.choose_game_mode()

    def choose_game_mode(self):
        """Shows the game mode selection window."""
        mode_window = tk.Toplevel(self.master)
        mode_window.title("Choose Game Mode")
        mode_window.configure(bg=BACKGROUND_COLOR)
        mode_window.geometry("400x250")
        mode_window.transient(self.master)
        mode_window.grab_set()

        # Center the mode selection window
        mode_window.update_idletasks()
        x = (self.master.winfo_screenwidth() - mode_window.winfo_width()) // 2
        y = (self.master.winfo_screenheight() - mode_window.winfo_height()) // 2
        mode_window.geometry(f"+{x}+{y}")

        # Make the window modal
        mode_window.focus_set()
        mode_window.protocol("WM_DELETE_WINDOW", lambda: None)  # Disable close button

        label = tk.Label(
            mode_window,
            text="Choose your game mode:",
            font=("Inter", 14, "bold"),
            fg="white",
            bg=BACKGROUND_COLOR,
            pady=10,
        )
        label.pack()

        # Button for Human vs Human
        human_vs_human_btn = tk.Button(
            mode_window,
            text="Human vs Human",
            command=lambda: self.set_game_mode("Human vs Human", mode_window),
            font=("Inter", 12),
            bg="#3498db",
            fg="black",
            width=20,
            pady=5,
        )
        human_vs_human_btn.pack(pady=5)

        # Button for Human vs Heuristic AI
        human_vs_heuristic_btn = tk.Button(
            mode_window,
            text="Human vs Heuristic AI",
            command=lambda: self.set_game_mode("Human vs Heuristic AI", mode_window),
            font=("Inter", 12),
            bg="#2ecc71",
            fg="black",
            width=20,
            pady=5,
        )
        human_vs_heuristic_btn.pack(pady=5)

        # Button for Human vs Neural Network AI
        human_vs_nn_btn = tk.Button(
            mode_window,
            text="Human vs Neural Network AI",
            command=lambda: self.set_game_mode(
                "Human vs Neural Network AI", mode_window
            ),
            font=("Inter", 12),
            bg="#9b59b6",
            fg="black",
            width=20,
            pady=5,
        )
        human_vs_nn_btn.pack(pady=5)

        # Info label
        info_label = tk.Label(
            mode_window,
            text="You can change mode anytime using the 'New Game' button",
            font=("Inter", 10),
            fg="#bdc3c7",
            bg=BACKGROUND_COLOR,
            pady=10,
        )
        info_label.pack()

    def set_game_mode(self, mode, mode_window):
        """Sets the game mode and closes the mode selection window."""
        self.game_mode = mode
        mode_window.destroy()
        self.start_new_game()
        messagebox.showinfo(
            "Game Mode Selected",
            f"You selected: {mode}\n\nYou are Player 1 (Red).\nClick on any column to drop your piece.",
        )

    def reset_game(self):
        """Shows the game mode selection screen to start a new game."""
        self.choose_game_mode()

    def start_new_game(self):
        """Starts a new game with the current game mode."""
        self.board = create_board()
        self.turn = 1
        self.game_over = False
        draw_board(self.canvas, self.board)
        if self.indicator_circle:
            self.canvas.delete(self.indicator_circle)
        self.update_status_label()

    def update_status_label(self):
        if self.game_mode == "Human vs Human":
            color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR
            self.status_label.config(
                text=f"Player {self.turn} ({color.capitalize()})'s Turn", fg="white"
            )
        else:
            if self.turn == 1:
                self.status_label.config(text="Your Turn (Red)", fg="white")
            else:
                self.status_label.config(text="AI's Turn (Yellow)", fg="white")

    def handle_motion(self, event):
        if self.game_over or self.game_mode is None:
            return

        col = max(0, min(COLUMN_COUNT - 1, event.x // SQUARE_SIZE))
        if 0 <= col < COLUMN_COUNT:
            if self.indicator_circle:
                self.canvas.delete(self.indicator_circle)

            x = col * SQUARE_SIZE + SQUARE_SIZE / 2
            y = SQUARE_SIZE / 2
            x1, y1, x2, y2 = x - RADIUS, y - RADIUS, x + RADIUS, y + RADIUS
            color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR

            if is_valid_location(self.board, col):
                self.indicator_circle = self.canvas.create_oval(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=color,
                    outline="#333",
                    width=2,
                    stipple="gray25",
                )

    def handle_click(self, event):
        if self.game_over:
            return

        col = int(math.floor(event.x / SQUARE_SIZE))
        if is_valid_location(self.board, col):
            self.play_move(col)

            # Trigger AI only if in AI mode and game not over
            if self.game_mode == "Human vs Heuristic AI" and not self.game_over:
                self.master.after(500, self.heuristic_ai_move)
            elif self.game_mode == "Human vs Neural Network AI" and not self.game_over:
                self.master.after(500, self.neural_network_ai_move)

    def play_move(self, col):
        row = get_next_open_row(self.board, col)
        drop_piece(self.board, row, col, self.turn)
        draw_board(self.canvas, self.board)

        if check_win(self.board, self.turn):
            self.game_over = True
            winner_color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR

            if self.game_mode == "Human vs Human":
                self.status_label.config(text=f"Player {self.turn} Wins!", fg="green")
                messagebox.showinfo(
                    "Game Over",
                    f"Player {self.turn} ({winner_color.capitalize()}) wins!",
                )
            else:
                if self.turn == 1:
                    self.status_label.config(text="You Win!", fg="green")
                    messagebox.showinfo("Game Over", "Congratulations! You win!")
                else:
                    self.status_label.config(text="AI Wins!", fg="red")
                    messagebox.showinfo("Game Over", "The AI wins!")

        elif is_board_full(self.board):
            self.game_over = True
            self.status_label.config(text="Game Drawn!", fg="orange")
            messagebox.showinfo("Game Over", "It's a draw!")
        else:
            self.turn = 3 - self.turn
            self.update_status_label()

    def heuristic_ai_move(self):
        """Call heuristic_search.get_best_move and play the returned column."""
        # choose ai_player = 2 (your GUI uses 2 for AI)
        ai_col = get_best_move(
            self.board, ai_player=2, max_depth=5
        )  # depth can be tuned
        if ai_col is None:
            # no valid heuristic move found — fallback to random
            valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(self.board, col)]
            if valid_cols:
                ai_col = random.choice(valid_cols)
            else:
                return
        # play the heuristic-selected column
        if is_valid_location(self.board, ai_col):
            self.play_move(ai_col)

    def neural_network_ai_move(self):
        """Use the trained neural network model to make a move."""
        if self.nn_model is None:
            # Fallback to random if model not available
            messagebox.showwarning(
                "Neural Network AI Unavailable",
                "Neural network model not loaded. Falling back to random moves."
            )
            valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(self.board, col)]
            if valid_cols:
                ai_col = random.choice(valid_cols)
                if is_valid_location(self.board, ai_col):
                    self.play_move(ai_col)
            return
        
        # AI is Player 2 in the GUI
        ai_player = 2
        BOARD_SIZE = ROW_COUNT * COLUMN_COUNT
        valid_cols = [col for col in range(COLUMN_COUNT) if is_valid_location(self.board, col)]
        
        if not valid_cols:
            return
        
        best_col = None
        best_win_prob = -1.0
        best_draw_prob = -1.0  # For tie-breaking
        
        for col in valid_cols:
            # Simulate the move
            temp_board = np.array([row[:] for row in self.board])
            row = get_next_open_row(temp_board, col)
            
            if row == -1:
                continue
            
            temp_board[row][col] = ai_player
            
            # Prepare board for NN: convert to model format
            # Model expects: 1.0 (Player 1), -1.0 (Player -1/Opponent), 0.0 (Empty)
            # Game uses: 1 (Player 1), 2 (AI), 0 (Empty)
            # Convert: 1 -> 1.0, 2 -> -1.0, 0 -> 0.0
            flat_board = temp_board.flatten().astype(float)
            nn_input_data = np.where(flat_board == 2.0, -1.0, flat_board).reshape(1, BOARD_SIZE)
            
            # Predict outcome: [P(Winner=-1), P(Draw), P(Winner=1)]
            # Since AI is Player 2 (which maps to -1.0), we want to maximize Index 0
            prediction = self.nn_model.predict(nn_input_data, verbose=0)[0]
            win_prob = prediction[0]  # Probability that Player -1 (our AI) wins
            draw_prob = prediction[1]  # Probability of a draw
            
            if win_prob > best_win_prob:
                best_win_prob = win_prob
                best_draw_prob = draw_prob
                best_col = col
            # Tie-breaker: if win probabilities are equal, prefer moves with higher draw probability
            elif win_prob == best_win_prob and draw_prob > best_draw_prob:
                best_draw_prob = draw_prob
                best_col = col
        
        # Make the move
        if best_col is not None and is_valid_location(self.board, best_col):
            self.play_move(best_col)
        else:
            # Fallback to first valid column if something went wrong
            if valid_cols:
                self.play_move(valid_cols[0])


if __name__ == "__main__":
    root = tk.Tk()
    screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
    app_width, app_height = WIDTH + 40, HEIGHT + 200
    x_cordinate = int((screen_width / 2) - (app_width / 2))
    y_cordinate = int((screen_height / 2) - (app_height / 2))
    root.geometry(f"{app_width}x{app_height}+{x_cordinate}+{y_cordinate}")
    root.resizable(True, True)
    game_app = ConnectFourApp(root)
    root.mainloop()
