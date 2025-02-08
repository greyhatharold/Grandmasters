"""
3D Chess Demo with LangChain-Powered AI and Animated Chat Log

This demo implements a hacked-up version of 3D chess on three levels (each level is an 8x8 board).
• Level 1 is initialized with white pieces (standard chess arrangement on ranks 1 and 2).
• Level 3 is initialized with black pieces (mirroring white’s arrangement on ranks 8 and 7).
• Level 2 starts empty.
Moves are represented in a custom notation:
    <from_level><from_square> - <to_level><to_square>
e.g., "1e2-1e4" (move within level 1) or "1e2-2e4" (move across levels).
  
A LangChain agent (backed by OpenAI’s Chat model) is used to generate moves.
The agent is instructed to output in the following strict format:

    MOVE: <move>
    THOUGHTS: <internal chain-of-thought and assumptions>

A chat log at the bottom shows the two models’ internal “conversations” (with animated text).
No user input is required; the game plays out automatically.

Requirements:
    - openai
    - langchain
    - tkinter (usually included with Python)
    
Install required packages:
    pip install openai langchain

Remember to set your OpenAI API key below.
"""

import tkinter as tk
from tkinter import scrolledtext
import threading, time, random, re
from dotenv import load_dotenv
import os
from typing import Tuple

# --- LangChain & OpenAI imports ---
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
import openai
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Define a LangChain tool for displaying board state (for agent’s “console access”) ---
def display_board_tool_3d(board_state: str) -> str:
    """
    This tool prints the 3D board state (for all levels) to the console.
    It is provided to the agent so it “knows” it can check the board.
    """
    print("\n=== Current 3D Chess Board State ===")
    print(board_state)
    print("====================================\n")
    return board_state

# --- Board3D Class ---
class Board3D:
    """
    A minimal 3D chess board representation.
    Levels: 1 (white's), 2 (empty), 3 (black's).
    Files: 'a' through 'h', Ranks: 1 through 8.
    Pieces are represented as two-letter strings:
      white: "wK", "wQ", "wR", "wB", "wN", "wP"
      black: "bK", "bQ", "bR", "bB", "bN", "bP"
    """
    files = "abcdefgh"
    
    def __init__(self):
        # Board is a dictionary with keys (level, file, rank)
        self.board = {}  
        self.current_turn = "w"  # white moves first
        self.setup_board()
    
    def setup_board(self):
        # Initialize white pieces on level 1:
        level = 1
        # Rooks, Knights, Bishops, Queen, King, Bishops, Knights, Rooks on rank 1
        pieces_order = ["wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"]
        for i, file in enumerate(Board3D.files):
            self.board[(level, file, 1)] = pieces_order[i]
        # Pawns on rank 2
        for file in Board3D.files:
            self.board[(level, file, 2)] = "wP"
        
        # Initialize black pieces on level 3:
        level = 3
        pieces_order = ["bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"]
        for i, file in enumerate(Board3D.files):
            self.board[(level, file, 8)] = pieces_order[i]
        for file in Board3D.files:
            self.board[(level, file, 7)] = "bP"
        # Level 2 remains empty.
    
    def to_string(self) -> str:
        """
        Return a string representation of all three levels.
        Each level is shown with ranks 8 (top) to 1 (bottom).
        """
        board_str = ""
        for level in [1, 2, 3]:
            board_str += f"Level {level}:\n"
            for rank in range(8, 0, -1):
                row = ""
                for file in Board3D.files:
                    piece = self.board.get((level, file, rank), ".")
                    # For a nicer display, show just one or two characters.
                    row += f"{piece:3}"  # fixed width
                board_str += row + "\n"
            board_str += "\n"
        return board_str

    def move_piece(self, move_str: str) -> bool:
        """
        Execute a move in the custom notation:
            <from_level><from_square>-<to_level><to_square>
        For example: "1e2-1e4" or "1e2-2e4".
        Returns True if move was executed; False if invalid.
        """
        pattern = r"^([123])([a-h][1-8])-([123])([a-h][1-8])$"
        m = re.match(pattern, move_str.strip())
        if not m:
            return False
        from_level = int(m.group(1))
        from_square = m.group(2)
        to_level = int(m.group(3))
        to_square = m.group(4)
        from_file, from_rank = from_square[0], int(from_square[1])
        to_file, to_rank = to_square[0], int(to_square[1])
        src = (from_level, from_file, from_rank)
        dst = (to_level, to_file, to_rank)
        # Check if source contains a piece belonging to the current turn.
        piece = self.board.get(src)
        if not piece or piece[0] != self.current_turn:
            return False
        # (No full legal-move validation is done; this is a hack!)
        # Execute move: capture any piece at destination.
        self.board.pop(src)
        self.board[dst] = piece
        # Toggle turn.
        self.current_turn = "b" if self.current_turn == "w" else "w"
        return True

    def is_game_over(self) -> bool:
        """
        Game is over if one of the kings is missing.
        """
        kings = [p for p in self.board.values() if p in ["wK", "bK"]]
        return len(kings) < 2

    def get_all_moves(self) -> list:
        """
        As a fallback, generate a list of possible moves for the current player.
        For each piece belonging to the current player, consider all moves that
        shift the piece by -1, 0, or +1 in level, file, and rank (except staying still).
        This is a very simplified “neighbor” move generator.
        """
        moves = []
        for (lvl, file, rank), piece in self.board.items():
            if piece[0] != self.current_turn:
                continue
            lvl_idx = lvl
            file_idx = Board3D.files.index(file)
            rank_val = rank
            # Consider 3D directions: dx, dy, dz in {-1, 0, 1} excluding all zeros.
            for dl in [-1, 0, 1]:
                for df in [-1, 0, 1]:
                    for dr in [-1, 0, 1]:
                        if dl == df == dr == 0:
                            continue
                        new_lvl = lvl_idx + dl
                        new_file_idx = file_idx + df
                        new_rank = rank_val + dr
                        if new_lvl < 1 or new_lvl > 3:
                            continue
                        if new_file_idx < 0 or new_file_idx > 7:
                            continue
                        if new_rank < 1 or new_rank > 8:
                            continue
                        from_sq = f"{file}{rank}"
                        to_sq = f"{Board3D.files[new_file_idx]}{new_rank}"
                        move_notation = f"{lvl}{from_sq}-{new_lvl}{to_sq}"
                        moves.append(move_notation)
        return moves

# --- Chess3DGUI Class ---
class Chess3DGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("3D Chess AI Duel with Animated Chat")
        self.running = True  # Flag for game loop

        # --- Set up frames ---
        self.board_frame = tk.Frame(root)
        self.board_frame.pack(side=tk.TOP, padx=10, pady=10)
        # We'll create three canvases (one per level), arranged vertically.
        self.canvas_size = 300  # each board will be 300x300 pixels
        self.canvases = {}  # key: level number, value: canvas widget
        for level in [1, 2, 3]:
            frame = tk.Frame(self.board_frame)
            frame.pack(pady=5)
            label = tk.Label(frame, text=f"Level {level}", font=("Helvetica", 14, "bold"))
            label.pack()
            canvas = tk.Canvas(frame, width=self.canvas_size, height=self.canvas_size, bg="white")
            canvas.pack()
            self.canvases[level] = canvas

        # --- Chat log (scrolled text) ---
        self.chat_log = scrolledtext.ScrolledText(root, wrap=tk.WORD, height=10, state=tk.DISABLED, font=("Courier", 10))
        self.chat_log.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # --- Initialize the 3D chess board ---
        self.board3d = Board3D()

        # --- Initialize LangChain LLM and Agent ---
        self.llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
        # Set up our tool that gives the agent access to the board’s printed state.
        tools = [
            Tool(
                name="display_board_3d",
                func=display_board_tool_3d,
                description=(
                    "Use this tool to display the current 3D chess board state. "
                    "Provide the board state (as produced by the Board3D.to_string() method) as input."
                )
            )
        ]
        # Use a zero-shot agent that can reason about using the tool.
        self.agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)

        # --- Draw initial boards ---
        self.draw_boards()

        # --- Start game loop in a separate thread ---
        self.game_thread = threading.Thread(target=self.game_loop, daemon=True)
        self.game_thread.start()

    def draw_boards(self):
        """Redraw all three level boards."""
        square_size = self.canvas_size // 8
        for level in [1, 2, 3]:
            canvas = self.canvases[level]
            canvas.delete("all")
            # Draw board squares (alternating colors)
            for r in range(8):
                for c in range(8):
                    x1 = c * square_size
                    y1 = r * square_size
                    x2 = x1 + square_size
                    y2 = y1 + square_size
                    # Alternate colors (simple check)
                    color = "#F0D9B5" if (r+c) % 2 == 0 else "#B58863"
                    canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
            # Draw pieces on this level
            for file in Board3D.files:
                for rank in range(1, 9):
                    pos = (level, file, rank)
                    piece = self.board3d.board.get(pos)
                    if piece:
                        # Center the text within the square.
                        c = Board3D.files.index(file)
                        r = 8 - rank  # invert so that rank 8 is at top
                        x = c * square_size + square_size // 2
                        y = r * square_size + square_size // 2
                        # For simplicity, just draw the two-letter piece code.
                        canvas.create_text(x, y, text=piece, font=("Helvetica", 16, "bold"))
        # Force update of the GUI
        self.root.update_idletasks()

    def append_chat(self, message: str):
        """Append a message to the chat log (without animation)."""
        self.chat_log.config(state=tk.NORMAL)
        self.chat_log.insert(tk.END, message + "\n")
        self.chat_log.see(tk.END)
        self.chat_log.config(state=tk.DISABLED)

    def animate_chat(self, message: str, delay: int = 20):
        """
        Animate the given message into the chat log, letter by letter.
        'delay' is in milliseconds.
        """
        self.chat_log.config(state=tk.NORMAL)
        def type_character(idx=0):
            if idx < len(message):
                self.chat_log.insert(tk.END, message[idx])
                self.chat_log.see(tk.END)
                self.root.after(delay, type_character, idx+1)
            else:
                self.chat_log.insert(tk.END, "\n")
                self.chat_log.see(tk.END)
                self.chat_log.config(state=tk.DISABLED)
        type_character()

    def get_langchain_move(self) -> tuple[str, str]:
        """
        Ask the LangChain agent for a move. The prompt instructs the agent to output exactly in the format:

            MOVE: <move in format like 1e2-1e4>
            THOUGHTS: <internal chain-of-thought and assumptions>

        Returns a tuple: (move_str, thoughts)
        """
        color = "White" if self.board3d.current_turn == "w" else "Black"
        board_state = self.board3d.to_string()
        prompt = (
            f"You are a 3D chess engine playing as {color}. The current board state is:\n\n"
            f"{board_state}\n\n"
            "You have access to a tool named 'display_board_3d' that can show the board state if needed. "
            "Determine your next move. Output exactly in the following format (and nothing else):\n\n"
            "MOVE: <move in the format like 1e2-1e4>\n"
            "THOUGHTS: <your internal chain-of-thought and assumptions>\n"
        )
        try:
            response = self.agent.run(prompt)
            # Use regex to extract MOVE and THOUGHTS from the response.
            move_match = re.search(r"MOVE:\s*([123][a-h][1-8]-[123][a-h][1-8])", response)
            thoughts_match = re.search(r"THOUGHTS:\s*(.*)", response, re.DOTALL)
            if move_match:
                move_str = move_match.group(1).strip()
            else:
                move_str = ""
            if thoughts_match:
                thoughts = thoughts_match.group(1).strip()
            else:
                thoughts = ""
            return move_str, thoughts
        except Exception as e:
            print("LangChain agent error:", e)
            return "", ""

    def game_loop(self):
        """
        Main game loop that alternates moves between the two AI agents.
        For each turn, it:
          1. Asks the LangChain agent for a move (and its internal thoughts).
          2. Validates the move (ensuring that the source square contains a piece for the current turn).
          3. If the move is invalid, falls back to a random move generated from nearby moves.
          4. Updates the board and redraws the 3D boards.
          5. Animates the chat log with the move and the agent’s internal thoughts.
        """
        while not self.board3d.is_game_over() and self.running:
            current_color = "White" if self.board3d.current_turn == "w" else "Black"
            move_str, thoughts = self.get_langchain_move()
            valid = self.board3d.move_piece(move_str)
            if not valid:
                # Log the invalid move and choose a fallback random move.
                fallback_moves = self.board3d.get_all_moves()
                if fallback_moves:
                    move_str = random.choice(fallback_moves)
                    self.board3d.move_piece(move_str)
                    fallback_note = " (fallback random move)"
                else:
                    move_str = "No valid moves"
                    fallback_note = ""
                thoughts = "Agent produced an invalid move. Using fallback move." + fallback_note
            # Create a chat message showing the agent's move and its internal chain-of-thought.
            chat_message = f"{current_color}:\nMove: {move_str}\nThoughts: {thoughts}\n{'-'*40}"
            # Schedule the animated chat message insertion on the main thread.
            self.root.after(0, self.animate_chat, chat_message)
            # Redraw the boards.
            self.root.after(0, self.draw_boards)
            # Pause between moves.
            time.sleep(2)
        # Game over message.
        final_state = "Game Over. "
        if self.board3d.is_game_over():
            final_state += "A king has been captured."
        else:
            final_state += "Game stopped."
        self.root.after(0, self.append_chat, final_state)

# --- Graceful shutdown function ---
def on_closing(gui: Chess3DGUI, root):
    gui.running = False
    root.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = Chess3DGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(gui, root))
    root.mainloop()