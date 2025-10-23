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
            text="",
            font=("Inter", 16, "bold"),
            fg="white",
            bg=BACKGROUND_COLOR,
        )
        self.status_label.pack(pady=10)

        # --- Reset Button ---
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

        # --- Bindings ---
        self.canvas.bind("<Button-1>", self.handle_click)
        self.canvas.bind("<Motion>", self.handle_motion)
        self.indicator_circle = None

        # --- Choose game mode at startup ---
        self.choose_game_mode()

    def choose_game_mode(self):
        """Prompts the user to choose whether to play against another human or AI."""
        response = messagebox.askquestion(
            "Choose Game Mode",
            "Would you like to play against the computer?",
            icon="question",
        )
        # 'yes' means play vs AI, 'no' means play vs another player
        if response == "yes":
            self.game_mode = "Human vs AI (Random)"
            messagebox.showinfo("Game Mode Selected", "You are playing against the AI.")
        else:
            self.game_mode = "Human vs Human"
            messagebox.showinfo(
                "Game Mode Selected", "You are playing against another player."
            )
        self.update_status_label()

    def reset_game(self):
        """Resets the board and game state, and reprompts mode selection."""
        self.board = create_board()
        self.turn = 1
        self.game_over = False
        draw_board(self.canvas, self.board)
        if self.indicator_circle:
            self.canvas.delete(self.indicator_circle)
        # Prompt again to allow changing modes between games
        self.choose_game_mode()

    def update_status_label(self):
        color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR
        self.status_label.config(
            text=f"Player {self.turn} ({color.capitalize()})'s Turn", fg="white"
        )

    def handle_motion(self, event):
        if self.game_over:
            return

        col = int(math.floor(event.x / SQUARE_SIZE))
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
            if self.game_mode == "Human vs AI (Random)" and not self.game_over:
                self.master.after(500, self.ai_move)

    def play_move(self, col):
        row = get_next_open_row(self.board, col)
        drop_piece(self.board, row, col, self.turn)
        draw_board(self.canvas, self.board)

        if check_win(self.board, self.turn):
            self.game_over = True
            winner_color = PLAYER1_COLOR if self.turn == 1 else PLAYER2_COLOR
            self.status_label.config(text=f"Player {self.turn} Wins!", fg="green")
            messagebox.showinfo(
                "Game Over",
                f"Player {self.turn} ({winner_color.capitalize()}) wins!",
            )
        elif is_board_full(self.board):
            self.game_over = True
            self.status_label.config(text="Game Drawn!", fg="orange")
            messagebox.showinfo("Game Over", "It's a draw!")
        else:
            self.turn = 3 - self.turn
            self.update_status_label()

    def ai_move(self):
        """Simple AI that plays a random valid column."""
        valid_cols = [
            c for c in range(COLUMN_COUNT) if is_valid_location(self.board, c)
        ]
        if not valid_cols:
            return
        ai_col = random.choice(valid_cols)
        self.play_move(ai_col)


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
