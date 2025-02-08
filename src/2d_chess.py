"""
Chess GUI with LangChain-Enabled AI Moves

This script creates a chess game where two AI agents (one playing White, one playing Black)
compete against each other. The chess logic is handled by python-chess and the graphical
interface is built using Tkinter. For move generation, we use LangChain to wrap the OpenAI
Chat API, providing the agent with a custom tool that prints the board state to the console.
This simulates giving the AI access to the console before deciding on its move.

Requirements:
    - python-chess
    - langchain
    - openai
    - tkinter (usually included with Python)

Installation (if needed):
    pip install python-chess langchain openai
"""

import tkinter as tk
import chess
import time
import threading
import random
import os
from dotenv import load_dotenv

# --- LangChain and OpenAI imports ---
from langchain_community.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool

# --- OpenAI API key configuration ---
import openai

# Set the API key as an environment variable
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# --- Define a LangChain Tool for console access ---
def display_board_tool(fen: str) -> str:
    """
    Given a FEN string, this function creates a python-chess Board object,
    prints the board state (using a Unicode representation) to the console,
    and returns the string representation.
    
    This tool is provided to the LangChain agent so that it can "see" the board
    state on the console if needed.
    """
    board = chess.Board(fen)
    board_str = board.unicode()  # Unicode representation of the board
    print("\n=== Current Board State (Console) ===")
    print(board_str)
    print("=====================================\n")
    return board_str

# --- Main Chess GUI Class ---
class ChessGUI:
    def __init__(self, master):
        # Initialize the main window
        self.master = master
        self.master.title("LangChain-Enabled OpenAI vs. OpenAI Chess")
        
        # Define canvas dimensions for the chessboard (8x8 squares)
        self.canvas_size = 480  # Total size in pixels (each square will be 60x60)
        self.square_size = self.canvas_size // 8
        self.canvas = tk.Canvas(self.master, width=self.canvas_size, height=self.canvas_size)
        self.canvas.pack()
        
        # Initialize the python-chess board
        self.board = chess.Board()
        
        # Colors for the board squares
        self.colors = ["#F0D9B5", "#B58863"]
        
        # Mapping from piece symbols to Unicode characters (for display)
        self.piece_unicode = {
            'P': "\u2659",  # White Pawn
            'R': "\u2656",  # White Rook
            'N': "\u2658",  # White Knight
            'B': "\u2657",  # White Bishop
            'Q': "\u2655",  # White Queen
            'K': "\u2654",  # White King
            'p': "\u265F",  # Black Pawn
            'r': "\u265C",  # Black Rook
            'n': "\u265E",  # Black Knight
            'b': "\u265D",  # Black Bishop
            'q': "\u265B",  # Black Queen
            'k': "\u265A"   # Black King
        }
        
        # Draw the initial board on the GUI canvas
        self.draw_board()
        
        # --- Set up LangChain LLM and Agent ---
        # Initialize the ChatOpenAI LLM with zero temperature for deterministic responses.
        self.llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")
        
        # Define a list of tools that the agent can use.
        # Here we only include the "display_board" tool, which gives the AI console access.
        tools = [
            Tool(
                name="display_board",
                func=display_board_tool,
                description=(
                    "Use this tool to display the current chess board state in the console. "
                    "Provide the board's FEN (Forsyth–Edwards Notation) as input. "
                    "This helps in visualizing the board when deciding on a move."
                )
            )
        ]
        
        # Initialize the LangChain agent with the tool.
        # The agent type 'zero-shot-react-description' allows it to reason about using the tool.
        self.agent = initialize_agent(tools, self.llm, agent="zero-shot-react-description", verbose=True)
        
        # A flag to allow graceful shutdown of the game loop
        self.running = True
        
        # Start the game loop in a separate thread so the GUI remains responsive.
        self.game_thread = threading.Thread(target=self.game_loop, daemon=True)
        self.game_thread.start()
    
    def draw_board(self):
        """
        Redraws the chessboard and all pieces.
        This function clears the canvas, redraws the colored squares,
        and overlays the chess pieces using Unicode symbols.
        """
        self.canvas.delete("all")  # Clear the canvas
        
        # Draw the squares of the chessboard.
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size
                y1 = (7 - rank) * self.square_size  # Invert rank for proper display
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                # Determine the color for this square
                color = self.colors[(file + rank) % 2]
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color, outline="")
        
        # Draw the chess pieces.
        for square in chess.SQUARES:
            piece = self.board.piece_at(square)
            if piece:
                piece_symbol = piece.symbol()
                # Get the Unicode character for the piece (or default to the symbol)
                piece_text = self.piece_unicode.get(piece_symbol, piece_symbol)
                file = chess.square_file(square)
                rank = chess.square_rank(square)
                x = file * self.square_size + self.square_size // 2
                y = (7 - rank) * self.square_size + self.square_size // 2
                self.canvas.create_text(x, y, text=piece_text, font=("Arial", 36))
        
        # Update the GUI window
        self.master.update()

    def game_loop(self):
        """
        The main game loop that continues until the game is over or the user closes the window.
        On each turn, the loop:
            1. Determines which side is to move.
            2. Uses the LangChain agent to generate a move (with console access provided by the tool).
            3. Validates the move (falling back to a random legal move if necessary).
            4. Updates the board state and GUI.
            5. Waits briefly before the next move.
        """
        while not self.board.is_game_over() and self.running:
            current_color = self.board.turn  # True for White, False for Black
            color_str = "White" if current_color == chess.WHITE else "Black"
            
            # Get the move using the LangChain-powered agent.
            move_uci = self.get_langchain_move(self.board, current_color)
            
            # Validate and parse the move.
            try:
                move = self.board.parse_uci(move_uci)
                if move not in self.board.legal_moves:
                    raise ValueError("Move is not legal.")
            except Exception as e:
                print(f"Agent returned an invalid move ({move_uci}): {e}")
                # Fallback to a random legal move.
                move = random.choice(list(self.board.legal_moves))
                print(f"Falling back to random move: {move.uci()}")
            
            # Push the move to the board and redraw the GUI.
            self.board.push(move)
            print(f"{color_str} moves: {move.uci()}")
            self.draw_board()
            
            # Pause briefly between moves.
            time.sleep(1)
        
        print("Game over:", self.board.result())

    def get_langchain_move(self, board, color):
        """
        Uses the LangChain agent (which has access to the console via the 'display_board' tool)
        to generate a move in UCI format.
        
        The prompt provided to the agent includes:
            - The color to move.
            - The current board state in FEN.
            - An instruction to output only the move in UCI format (e.g., e2e4).
            - A note that the 'display_board' tool is available to help display the board.
        
        If the agent's response contains extra text, only the first token is extracted.
        """
        prompt = (
            f"You are a chess engine playing as {'White' if color == chess.WHITE else 'Black'}. "
            f"The current board state (in FEN) is:\n\n{board.fen()}\n\n"
            "You have access to a console tool named 'display_board' that can show the board state if needed. "
            "Your task is to decide your move. Output only the move in UCI format (e.g., e2e4) without any additional commentary."
        )
        
        try:
            # Run the prompt through the LangChain agent.
            response = self.agent.run(prompt)
            # Extract the first token in case the response includes extra text.
            move_text = response.strip().split()[0]
            print(f"LangChain agent ({'White' if color == chess.WHITE else 'Black'}) suggests: {move_text}")
            return move_text
        except Exception as e:
            print("LangChain agent error:", e)
            # In case of error, return a random legal move.
            return random.choice([m.uci() for m in board.legal_moves])

# --- Function to gracefully handle window closing ---
def on_closing(gui):
    """
    When the window is closed, set the running flag to False to stop the game loop
    and then destroy the Tkinter window.
    """
    gui.running = False
    root.destroy()

# --- Main Program Execution ---
if __name__ == "__main__":
    root = tk.Tk()
    gui = ChessGUI(root)
    root.protocol("WM_DELETE_WINDOW", lambda: on_closing(gui))
    root.mainloop()
