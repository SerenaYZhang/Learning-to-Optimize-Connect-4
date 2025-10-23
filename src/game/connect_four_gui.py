import tkinter as tk
from tkinter import messagebox
import math
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

# --- GUI Constants ---
SQUARE_SIZE = 100
RADIUS = int(SQUARE_SIZE / 2 - 5)
WIDTH = COLUMN_COUNT * SQUARE_SIZE
HEIGHT = (ROW_COUNT + 1) * SQUARE_SIZE  # Extra row for the drop area/indicator
PLAYER1_COLOR = "red"
PLAYER2_COLOR = "yellow"
BOARD_COLOR = "#3498db"  # Connect 4 blue
EMPTY_COLOR = "white"
BACKGROUND_COLOR = "#2c3e50"

# --- Drawing Function ---


def draw_board(canvas, board):
    """Draws the Connect 4 board and all current pieces on the canvas."""

    # 1. Draw the blue board rectangle
    canvas.create_rectangle(
        0, SQUARE_SIZE, WIDTH, HEIGHT, fill=BOARD_COLOR, outline=BOARD_COLOR
    )

    # 2. Draw the slots and pieces
    for c in range(COLUMN_COUNT):
        for r in range(ROW_COUNT):
            # Calculate center coordinates for the circle
            x = c * SQUARE_SIZE + SQUARE_SIZE / 2

            # Y-coordinate calculation must account for Tkinter's top-down indexing (0 is top)
            # We want board[0] (bottom row) to be drawn near the bottom of the canvas (HEIGHT - SQUARE_SIZE)
            # The canvas drawing area for the board starts at y=SQUARE_SIZE (below the indicator row)
            y = HEIGHT - (r * SQUARE_SIZE + SQUARE_SIZE / 2)

            # Define circle bounding box
            x1 = x - RADIUS
            y1 = y - RADIUS
            x2 = x + RADIUS
            y2 = y + RADIUS

            fill_color = EMPTY_COLOR
            if board[r][c] == 1:
                fill_color = PLAYER1_COLOR
            elif board[r][c] == 2:
                fill_color = PLAYER2_COLOR

            # Draw the circle with a shadow effect for depth
            canvas.create_oval(x1, y1, x2, y2, fill=fill_color, outline="#333", width=2)
    canvas.update()


# --- Main Application Class ---


class ConnectFourApp:
    def __init__(self, master):
        self.master = master
        master.title("Connect 4")
        master.configure(bg=BACKGROUND_COLOR)

        self.board = create_board()
        self.turn = 1  # 1: Player 1 (Red), 2: Player 2 (Yellow)
        self.game_over = False

        # --- Setup Canvas ---
        self.canvas = tk.Canvas(
            master,
            width=WIDTH,
            height=HEIGHT,
            bg=BACKGROUND_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(pady=20, padx=20)

        # Draw the initial empty board
        draw_board(self.canvas, self.board)

        # --- Status Label ---
        self.status_label = tk.Label(
            master,
            text=f"Player 1 ({PLAYER1_COLOR.capitalize()})'s Turn",
            font=("Inter", 16, "bold"),
            fg="white",
            bg=BACKGROUND_COLOR,
        )
        self.status_label.pack(pady=10)

        # --- Game Mode & Reset Button ---
        # NOTE: This will later include options for Human vs. Human, Human vs. Heuristic, etc.
        self.reset_button = tk.Button(
            master,
            text="Reset Game",
            command=self.reset_game,
            font=("Inter", 12),
            bg="#e74c3c",
            fg="white",
            activebackground="#c0392b",
        )
        self.reset_button.pack(pady=5)

        # --- Bind Events ---
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_motion)  # For move indicator

        # Indicator circle (drawn in the top, non-board row)
        self.indicator_circle = None

    def reset_game(self):
        """Resets the board and game state."""
        self.board = create_board()
        self.turn = 1
        self.game_over = False
        draw_board(self.canvas, self.board)
        self.status_label.config(
            text=f"Player 1 ({PLAYER1_COLOR.capitalize()})'s Turn", fg="white"
        )
        if self.indicator_circle:
            self.canvas.delete(self.indicator_circle)

    def handle_motion(self, event):
        """Updates the indicator circle position as the mouse moves."""
        if self.game_over:
            return

        col = int(math.floor(event.x / SQUARE_SIZE))

        if 0 <= col < COLUMN_COUNT:
            # Delete previous indicator circle
            if self.indicator_circle:
                self.canvas.delete(self.indicator_circle)

            # Calculate position in the top, indicator row (y=SQUARE_SIZE/2)
            x = col * SQUARE_SIZE + SQUARE_SIZE / 2
            y = SQUARE_SIZE / 2  # The center of the top row

            x1 = x - RADIUS
            y1 = y - RADIUS
            x2 = x + RADIUS
            y2 = y + RADIUS

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
            else:
                self.indicator_circle = None  # Don't show indicator if column is full

    def handle_click(self, event):
        """Processes a mouse click event for human player moves."""
        if self.game_over:
            return

        col = int(math.floor(event.x / SQUARE_SIZE))

        if is_valid_location(self.board, col):
            # 1. Get row and drop piece using imported logic
            row = get_next_open_row(self.board, col)
            drop_piece(self.board, row, col, self.turn)

            # Redraw board
            draw_board(self.canvas, self.board)

            # 2. Check for win or draw
            if check_win(self.board, self.turn):
                self.game_over = True
                winner_color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR
                self.status_label.config(text=f"Player {self.turn} Wins!", fg="green")
                messagebox.showinfo(
                    "Game Over",
                    f"Player {self.turn} ({winner_color.capitalize()}) is the winner!",
                )
            elif is_board_full(self.board):
                self.game_over = True
                self.status_label.config(text="Game Drawn!", fg="orange")
                messagebox.showinfo("Game Over", "The board is full. It's a draw!")
            else:
                # 3. Switch turns
                self.turn = 3 - self.turn  # Toggles between 1 and 2
                next_player_color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR
                self.status_label.config(
                    text=f"Player {self.turn} ({next_player_color.capitalize()})'s Turn",
                    fg="white",
                )

                # Update indicator to reflect new player color
                self.handle_motion(event)

        # NOTE: AI integration will go here, checking if the new turn belongs to an AI
        # if self.turn == 2 and self.game_mode == "H_vs_AI":
        #     self.ai_move()


if __name__ == "__main__":
    # Initialize the main Tkinter window
    root = tk.Tk()

    # Calculate the required screen geometry to center the window
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()

    app_width = WIDTH + 40
    app_height = HEIGHT + 140

    x_cordinate = int((screen_width / 2) - (app_width / 2))
    y_cordinate = int((screen_height / 2) - (app_height / 2))

    root.geometry(f"{app_width}x{app_height}+{x_cordinate}+{y_cordinate}")
    root.resizable(True, True)

    # Create and run the application instance
    game_app = ConnectFourApp(root)
    root.mainloop()
