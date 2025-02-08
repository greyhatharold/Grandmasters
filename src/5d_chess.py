#!/usr/bin/env python3
"""
Themed 5D Chess with Multiverse Time Travel – Enhanced Rendering and Chain-of-Thought

This single-file prototype demonstrates:
 • A 5D chess game engine with multiverse time travel logic.
 • Two LLM players (using ChatGPT API calls) that are prompted to include detailed chain-of-thought reasoning.
 • A richly themed chess board rendered with PySide6's QOpenGLWidget, including wood‑grain gradients,
   square borders, and animated move highlights.
 • A chat log widget that displays the chain-of-thought and moves in a nicely formatted HTML style.

Note: In production, you would modularize these components. For testing, ensure you have installed:
    pip install PySide6 openai
And replace "your-openai-api-key" with your actual API keys.
"""

import sys
import threading
import time
import re
import os
import math
import random

# PySide6 imports for GUI and OpenGL rendering
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QHBoxLayout, QTextEdit
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import QTimer, QRectF, Qt, QPointF
from PySide6.QtGui import QPainter, QPen, QColor, QFont, QLinearGradient, QPolygonF, QBrush, QPainterPath

# OpenAI API integration
import openai
from langchain.agents import create_react_agent, AgentExecutor
from langchain.prompts import PromptTemplate
from langchain.agents import Tool
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# =============================================================================
# High-Level: Chess Rules and Move Validation for 5D Multiverse Chess
# =============================================================================

class ChessRules:
    """Defines all valid moves and rules for 5D Multiverse Chess."""
    
    # Timeline Constants
    MAX_TIMELINES = 5
    MIN_MOVES_BEFORE_TIME_TRAVEL = 1  # Must make at least 1 move before time travel
    MAX_BRANCHING_FACTOR = 2  # Maximum number of branches from a single timeline
    
    # Piece movement vectors
    KNIGHT_MOVES = [(2,1), (2,-1), (-2,1), (-2,-1), (1,2), (1,-2), (-1,2), (-1,-2)]
    BISHOP_MOVES = [(1,1), (1,-1), (-1,1), (-1,-1)]
    ROOK_MOVES = [(1,0), (-1,0), (0,1), (0,-1)]
    QUEEN_MOVES = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    KING_MOVES = [(1,0), (-1,0), (0,1), (0,-1), (1,1), (1,-1), (-1,1), (-1,-1)]
    
    @staticmethod
    def is_valid_square(rank, file):
        """Check if a square is within the board boundaries."""
        return 0 <= rank < 8 and 0 <= file < 8
    
    @staticmethod
    def get_piece_color(piece):
        """Return the color of a piece ('white' for uppercase, 'black' for lowercase)."""
        return 'white' if piece.isupper() else 'black'
    
    @staticmethod
    def is_valid_move(board, from_square, to_square):
        """Check if a move is valid according to chess rules."""
        from_rank = 8 - int(from_square[1])
        from_file = ord(from_square[0]) - ord('a')
        to_rank = 8 - int(to_square[1])
        to_file = ord(to_square[0]) - ord('a')
        
        # Get the piece being moved
        piece = board[from_rank][from_file]
        if not piece:
            return False, "No piece at source square"
        
        # Get the target square's piece (if any)
        target = board[to_rank][to_file]
        if target and ChessRules.get_piece_color(target) == ChessRules.get_piece_color(piece):
            return False, "Cannot capture own piece"
        
        # Check piece-specific rules
        piece_type = piece.upper()
        if piece_type == 'P':
            return ChessRules.is_valid_pawn_move(board, from_rank, from_file, to_rank, to_file, piece)
        elif piece_type == 'N':
            return ChessRules.is_valid_knight_move(from_rank, from_file, to_rank, to_file)
        elif piece_type == 'B':
            return ChessRules.is_valid_bishop_move(board, from_rank, from_file, to_rank, to_file)
        elif piece_type == 'R':
            return ChessRules.is_valid_rook_move(board, from_rank, from_file, to_rank, to_file)
        elif piece_type == 'Q':
            return ChessRules.is_valid_queen_move(board, from_rank, from_file, to_rank, to_file)
        elif piece_type == 'K':
            return ChessRules.is_valid_king_move(board, from_rank, from_file, to_rank, to_file)
        return False, "Invalid piece type"
    
    @staticmethod
    def is_valid_pawn_move(board, from_rank, from_file, to_rank, to_file, piece):
        """Check if a pawn move is valid."""
        direction = -1 if piece.isupper() else 1  # White moves up (-1), Black moves down (+1)
        
        # Basic one square forward move
        if from_file == to_file and to_rank == from_rank + direction:
            return not board[to_rank][to_file]
        
        # Initial two square move
        if from_file == to_file and ((piece.isupper() and from_rank == 6 and to_rank == 4) or
                                    (piece.islower() and from_rank == 1 and to_rank == 3)):
            return (not board[to_rank][to_file] and 
                   not board[from_rank + direction][from_file])
        
        # Capture moves
        if to_rank == from_rank + direction and abs(to_file - from_file) == 1:
            return board[to_rank][to_file] and ChessRules.get_piece_color(board[to_rank][to_file]) != ChessRules.get_piece_color(piece)
        
        return False
    
    @staticmethod
    def is_valid_knight_move(from_rank, from_file, to_rank, to_file):
        """Check if a knight move is valid."""
        diff = (to_rank - from_rank, to_file - from_file)
        return diff in ChessRules.KNIGHT_MOVES
    
    @staticmethod
    def is_valid_bishop_move(board, from_rank, from_file, to_rank, to_file):
        """Check if a bishop move is valid."""
        if abs(to_rank - from_rank) != abs(to_file - from_file):
            return False
        
        rank_step = 1 if to_rank > from_rank else -1
        file_step = 1 if to_file > from_file else -1
        
        current_rank = from_rank + rank_step
        current_file = from_file + file_step
        
        while current_rank != to_rank:
            if board[current_rank][current_file]:
                return False
            current_rank += rank_step
            current_file += file_step
        
        return True
    
    @staticmethod
    def is_valid_rook_move(board, from_rank, from_file, to_rank, to_file):
        """Check if a rook move is valid."""
        if from_rank != to_rank and from_file != to_file:
            return False
        
        if from_rank == to_rank:
            step = 1 if to_file > from_file else -1
            for file in range(from_file + step, to_file, step):
                if board[from_rank][file]:
                    return False
        else:
            step = 1 if to_rank > from_rank else -1
            for rank in range(from_rank + step, to_rank, step):
                if board[rank][from_file]:
                    return False
        
        return True
    
    @staticmethod
    def is_valid_queen_move(board, from_rank, from_file, to_rank, to_file):
        """Check if a queen move is valid."""
        return (ChessRules.is_valid_bishop_move(board, from_rank, from_file, to_rank, to_file) or
                ChessRules.is_valid_rook_move(board, from_rank, from_file, to_rank, to_file))
    
    @staticmethod
    def is_valid_king_move(board, from_rank, from_file, to_rank, to_file):
        """Check if a king move is valid."""
        rank_diff = abs(to_rank - from_rank)
        file_diff = abs(to_file - from_file)
        return rank_diff <= 1 and file_diff <= 1
    
    @staticmethod
    def is_valid_timeline_creation(game_state, source_timeline, move, turn_number):
        """Check if a timeline creation is valid according to 5D chess rules."""
        # Rule 1: Can't exceed maximum number of timelines
        if len(game_state.timelines) >= ChessRules.MAX_TIMELINES:
            return False, "Maximum number of timelines reached"
        
        # Rule 2: Source timeline must exist
        if source_timeline not in game_state.timelines:
            return False, "Invalid source timeline"
        
        # Rule 3: Must make minimum moves before time travel
        if turn_number < ChessRules.MIN_MOVES_BEFORE_TIME_TRAVEL:
            return False, f"Must make at least {ChessRules.MIN_MOVES_BEFORE_TIME_TRAVEL} moves before time travel"
        
        # Rule 4: Check branching factor limit
        branch_count = sum(1 for hist in game_state.history 
                         if "time_travel" in str(hist[1]) and 
                         str(hist[1]).split()[-1][1:] == str(source_timeline))
        if branch_count >= ChessRules.MAX_BRANCHING_FACTOR:
            return False, f"Timeline {source_timeline} has reached maximum branches"
        
        # Rule 5: Move must be valid in source timeline
        board = game_state.timelines[source_timeline]
        move_valid, move_error = ChessRules.is_valid_move(board, move[0:2], move[2:4])
        if not move_valid:
            return False, f"Invalid move in source timeline: {move_error}"
        
        # Rule 6: Can't create timeline that would result in immediate checkmate
        new_board = [row[:] for row in board]
        from_rank = 8 - int(move[1])
        from_file = ord(move[0]) - ord('a')
        to_rank = 8 - int(move[3])
        to_file = ord(move[2]) - ord('a')
        piece = new_board[from_rank][from_file]
        new_board[from_rank][from_file] = ''
        new_board[to_rank][to_file] = piece
        
        if ChessRules.is_checkmate(new_board):
            return False, "Cannot create timeline resulting in immediate checkmate"
        
        return True, "Timeline creation is valid"
    
    @staticmethod
    def get_timeline_branches(game_state, timeline_id):
        """Get all timelines that branch from a given timeline."""
        branches = []
        for hist_timeline, move, _ in game_state.history:
            if "time_travel" in str(move):
                source = int(move.split()[-1][1:])  # Extract source timeline number
                if source == timeline_id:
                    branches.append(hist_timeline)
        return branches
    
    @staticmethod
    def can_move_to_timeline(game_state, current_timeline, target_timeline):
        """Check if a piece can move to a different timeline."""
        # Rule: Can only move to directly connected timelines
        if target_timeline not in ChessRules.get_timeline_branches(game_state, current_timeline):
            return False, "Can only move to directly connected timelines"
        return True, "Timeline move is valid"
    
    @staticmethod
    def get_valid_time_travel_moves(game_state, timeline_id, is_white_turn, turn_number):
        """Get all valid time travel moves from a given timeline."""
        valid_moves = []
        regular_moves = ChessRules.get_valid_moves(game_state, timeline_id, is_white_turn)
        
        for move in regular_moves:
            is_valid, reason = ChessRules.is_valid_timeline_creation(
                game_state, timeline_id, move, turn_number
            )
            if is_valid:
                valid_moves.append(f"time_travel {move} t{timeline_id}")
        
        return valid_moves, "Valid time travel moves retrieved"
    
    @staticmethod
    def validate_timeline_state(game_state):
        """Validate the overall state of all timelines."""
        errors = []
        
        # Check 1: Number of timelines
        if len(game_state.timelines) > ChessRules.MAX_TIMELINES:
            errors.append(f"Too many timelines: {len(game_state.timelines)}/{ChessRules.MAX_TIMELINES}")
        
        # Check 2: Branching factors
        for timeline_id in game_state.timelines:
            branches = ChessRules.get_timeline_branches(game_state, timeline_id)
            if len(branches) > ChessRules.MAX_BRANCHING_FACTOR:
                errors.append(f"Timeline {timeline_id} has too many branches: {len(branches)}/{ChessRules.MAX_BRANCHING_FACTOR}")
        
        # Check 3: Timeline connectivity (no orphaned timelines)
        for timeline_id in game_state.timelines:
            if timeline_id != 0:  # Skip main timeline
                is_connected = False
                for _, move, _ in game_state.history:
                    if "time_travel" in str(move) and str(timeline_id) in move:
                        is_connected = True
                        break
                if not is_connected:
                    errors.append(f"Timeline {timeline_id} is orphaned")
        
        return len(errors) == 0, errors

# =============================================================================
# High-Level: Game State Management for 5D (Multiverse) Chess
# =============================================================================

class GameState:
    def __init__(self):
        # Each timeline holds a board state; history records (timeline, move, chain-of-thought)
        self.timelines = {}
        self.history = []
        self.timeline_positions = {}  # Stores visual positions of timelines
        self.active_timeline = 0
        # Initialize main timeline with id 0
        self.timelines[0] = self.initialize_board()
        self.timeline_positions[0] = {'x': 0, 'y': 0, 'angle': 0}
    
    def initialize_board(self):
        """Create a standard 8x8 board.
           (For a full 5D game, pieces would have extra dimensions.)
        """
        board = [['' for _ in range(8)] for _ in range(8)]
        # Place black pieces on top
        board[0] = ['r', 'n', 'b', 'q', 'k', 'b', 'n', 'r']
        board[1] = ['p'] * 8
        # Place white pieces on bottom
        board[6] = ['P'] * 8
        board[7] = ['R', 'N', 'B', 'Q', 'K', 'B', 'N', 'R']
        return board

    def apply_move(self, board, from_square, to_square):
        """Apply a move to the board, moving the piece from source to destination."""
        from_rank = 8 - int(from_square[1])
        from_file = ord(from_square[0]) - ord('a')
        to_rank = 8 - int(to_square[1])
        to_file = ord(to_square[0]) - ord('a')
        
        # Move the piece
        piece = board[from_rank][from_file]
        board[from_rank][from_file] = ''
        board[to_rank][to_file] = piece
        return board

    def make_move(self, timeline_id, move, cot=None):
        """Update the board state based on the move."""
        self.history.append((timeline_id, move, cot))
        
        if "time_travel" in move:
            # Parse time travel move format: "time_travel e2e4 t1" means move e2e4 and create timeline from t1
            parts = move.split()
            source_timeline = int(parts[-1][1:])  # Extract timeline number after 't'
            move_part = parts[1]  # The actual move part (e.g., "e2e4")
            
            # Create new timeline branching from the source timeline
            new_timeline = max(self.timelines.keys()) + 1
            self.timelines[new_timeline] = [row[:] for row in self.timelines[source_timeline]]
            
            # Calculate new timeline position (diagonal offset from source)
            source_pos = self.timeline_positions[source_timeline]
            self.timeline_positions[new_timeline] = {
                'x': source_pos['x'] + 100,  # Offset right
                'y': source_pos['y'] + 100,  # Offset down
                'angle': source_pos['angle'] - 15  # Rotate slightly
            }
            
            # Apply the move in the new timeline
            if len(move_part) >= 4:
                from_square = move_part[0:2]
                to_square = move_part[2:4]
                self.timelines[new_timeline] = self.apply_move(self.timelines[new_timeline], 
                                                             from_square, to_square)
            return new_timeline
        else:
            # Regular move
            move = move.replace(" ", "")  # Remove any spaces
            if len(move) >= 4:
                from_square = move[0:2]
                to_square = move[2:4]
                board = self.timelines[timeline_id]
                self.timelines[timeline_id] = self.apply_move(board, from_square, to_square)
        
        return timeline_id

    def get_board(self, timeline_id):
        """Return the board for a given timeline."""
        return self.timelines.get(timeline_id, None)

# =============================================================================
# High-Level: LLM Integration with Chain-of-Thought Prompting
# =============================================================================

class LLMPlayer:
    def __init__(self, name, api_key, renderer=None):
        self.name = name
        self.api_key = api_key
        self.renderer = renderer
        openai.api_key = api_key
        self.conversation_history = []
        
        # Update to use new import path for ChatOpenAI
        from langchain_openai import ChatOpenAI
        self.llm = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo")  # Changed to match 2D chess
        
        # Simplified tools list to match 2D chess approach
        self.tools = [
            Tool(
                name="display_board",
                func=self.display_board_tool,
                description=(
                    "Use this tool to display the current chess board state in the console. "
                    "Provide the timeline number (e.g., '0' for main timeline) as input. "
                    "This helps in visualizing the board when deciding on a move."
                )
            ),
            Tool(
                name="list_timelines",
                func=self.list_timelines_tool,
                description=(
                    "Get information about all active timelines. "
                    "No input needed, just pass an empty string."
                )
            )
        ]
        
        # Initialize the LangChain agent with simpler configuration
        from langchain.agents import initialize_agent
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="zero-shot-react-description",
            verbose=True
        )
    
    def display_board_tool(self, timeline_id_str):
        """Tool for displaying the current board state of a timeline."""
        try:
            timeline_id = int(str(timeline_id_str).strip().replace('"', '').replace("'", ""))
            board = self.game_state.get_board(timeline_id)
            if board:
                board_str = ["\n=== Current Board State (Timeline {}) ===".format(timeline_id)]
                for row in board:
                    board_str.append(" ".join(piece if piece else "." for piece in row))
                board_str.append("=====================================\n")
                result = "\n".join(board_str)
                print(result)  # Print to console for debugging
                return result
            return f"Invalid timeline: {timeline_id}"
        except ValueError:
            return "Error: Please provide just the timeline number (e.g., 0 for main timeline)"
        except Exception as e:
            return f"Error displaying board: {str(e)}"
    
    def list_timelines_tool(self, *args):
        """Tool for displaying information about all timelines."""
        try:
            timelines = self.game_state.timelines
            result = ["Active Timelines:"]
            for timeline_id in timelines:
                branches = ChessRules.get_timeline_branches(self.game_state, timeline_id)
                result.append(f"\nTimeline {timeline_id}:")
                result.append(f"  Active: {'Yes' if timeline_id == self.game_state.active_timeline else 'No'}")
                result.append(f"  Branches: {branches if branches else 'None'}")
            return "\n".join(result)
        except Exception as e:
            return f"Error listing timelines: {str(e)}"
    
    def is_valid_5d_move(self, board, move, color, timeline_id):
        """Validate a move according to 5D chess rules."""
        if len(move) < 4:
            return False, "Move too short"
            
        # Handle time travel moves
        if move.startswith("time_travel"):
            parts = move.split()
            if len(parts) != 3:
                return False, "Invalid time travel format"
            move = parts[1]
            source_timeline = int(parts[2][1:])
            if source_timeline >= len(self.game_state.timelines):
                return False, "Invalid timeline"
        
        # Parse the move
        from_file = ord(move[0].lower()) - ord('a')
        from_rank = 8 - int(move[1])
        to_file = ord(move[2].lower()) - ord('a')
        to_rank = 8 - int(move[3])
        
        # Basic bounds checking
        if not (0 <= from_file <= 7 and 0 <= from_rank <= 7 and 
                0 <= to_file <= 7 and 0 <= to_rank <= 7):
            return False, "Move out of bounds"
        
        # Get the piece being moved
        piece = board[from_rank][from_file]
        if not piece:
            return False, "No piece at source square"
        
        # Check piece color
        piece_color = 'white' if piece.isupper() else 'black'
        if (color == 'White' and not piece.isupper()) or (color == 'Black' and piece.isupper()):
            return False, "Wrong color piece"
        
        # Calculate movement deltas
        dx = abs(to_file - from_file)  # x-axis (files)
        dy = abs(to_rank - from_rank)  # y-axis (ranks)
        
        # Piece-specific movement validation
        piece_type = piece.upper()
        if piece_type == 'P':  # Pawn
            # Normal move
            if dx == 0:
                if (piece.isupper() and from_rank - to_rank == 1) or \
                   (piece.islower() and to_rank - from_rank == 1):
                    return True, "Valid pawn move"
                if ((piece.isupper() and from_rank == 6 and from_rank - to_rank == 2) or 
                    (piece.islower() and from_rank == 1 and to_rank - from_rank == 2)):
                    return True, "Valid pawn double move"
            # Capture
            elif dx == 1 and ((piece.isupper() and from_rank - to_rank == 1) or 
                             (piece.islower() and to_rank - from_rank == 1)):
                target = board[to_rank][to_file]
                if target and ((piece.isupper() and target.islower()) or 
                             (piece.islower() and target.isupper())):
                    return True, "Valid pawn capture"
            return False, "Invalid pawn move"
            
        elif piece_type == 'N':  # Knight
            if (dx == 2 and dy == 1) or (dx == 1 and dy == 2):
                return True, "Valid knight move"
            return False, "Invalid knight move"
            
        elif piece_type == 'B':  # Bishop
            if dx == dy:  # Must move equally in both axes
                # Check path
                step_x = 1 if to_file > from_file else -1
                step_y = 1 if to_rank > from_rank else -1
                x, y = from_file + step_x, from_rank + step_y
                while x != to_file:
                    if board[y][x]:
                        return False, "Path blocked"
                    x += step_x
                    y += step_y
                return True, "Valid bishop move"
            return False, "Invalid bishop move"
            
        elif piece_type == 'R':  # Rook
            if dx == 0 or dy == 0:  # Must move along one axis
                if dx == 0:  # Vertical move
                    step = 1 if to_rank > from_rank else -1
                    for y in range(from_rank + step, to_rank, step):
                        if board[y][from_file]:
                            return False, "Path blocked"
                else:  # Horizontal move
                    step = 1 if to_file > from_file else -1
                    for x in range(from_file + step, to_file, step):
                        if board[from_rank][x]:
                            return False, "Path blocked"
                return True, "Valid rook move"
            return False, "Invalid rook move"
            
        elif piece_type == 'Q':  # Queen
            if dx == dy or dx == 0 or dy == 0:  # Can move along any axis
                if dx == dy:  # Diagonal move
                    step_x = 1 if to_file > from_file else -1
                    step_y = 1 if to_rank > from_rank else -1
                    x, y = from_file + step_x, from_rank + step_y
                    while x != to_file:
                        if board[y][x]:
                            return False, "Path blocked"
                        x += step_x
                        y += step_y
                elif dx == 0:  # Vertical move
                    step = 1 if to_rank > from_rank else -1
                    for y in range(from_rank + step, to_rank, step):
                        if board[y][from_file]:
                            return False, "Path blocked"
                else:  # Horizontal move
                    step = 1 if to_file > from_file else -1
                    for x in range(from_file + step, to_file, step):
                        if board[from_rank][x]:
                            return False, "Path blocked"
                return True, "Valid queen move"
            return False, "Invalid queen move"
            
        elif piece_type == 'K':  # King
            if dx <= 1 and dy <= 1:
                return True, "Valid king move"
            return False, "Invalid king move"
        
        return False, "Unknown piece type"

    def get_move(self, game_state):
        """Get a move from the LangChain agent, matching 2D chess implementation."""
        self.game_state = game_state
        
        if self.renderer:
            self.renderer.clear_thinking_visualization()
        
        try:
            # Prepare the prompt similar to 2D chess but with timeline support
            color = "White" if len(game_state.history) % 2 == 0 else "Black"
            timeline_id = game_state.active_timeline
            board = game_state.get_board(timeline_id)
            
            # Build a board string representation with piece positions
            board_str = "\n".join(" ".join(piece if piece else "." for piece in row) for row in board)
            
            # Add game history with analysis
            history_str = "\nPrevious moves and their effects:\n"
            for i, (t, m, _) in enumerate(game_state.history[-5:]):
                history_str += f"{i+1}. Timeline {t}: {m}\n"
            
            # Build a detailed board analysis
            pieces = {
                'P': [], 'p': [], 'N': [], 'n': [], 'B': [], 'b': [],
                'R': [], 'r': [], 'Q': [], 'q': [], 'K': [], 'k': []
            }
            for rank in range(8):
                for file in range(8):
                    piece = board[rank][file]
                    if piece:
                        pieces[piece].append((rank, file))
            
            # Create a strategic analysis
            analysis = "\nBoard Analysis:\n"
            analysis += f"- Your pieces ({color}):\n"
            piece_symbols = "PNBRQK" if color == "White" else "pnbrqk"
            for p in piece_symbols:
                positions = pieces[p]
                if positions:
                    analysis += f"  {p}: at {', '.join(f'{chr(97+pos[1])}{8-pos[0]}' for pos in positions)}\n"
            
            prompt = (
                f"You are a creative chess engine playing as {color} in timeline {timeline_id}. "
                f"The current board state is:\n\n{board_str}\n"
                f"{history_str}\n"
                f"{analysis}\n"
                "You have access to tools named 'display_board' and 'list_timelines' that can show more information if needed.\n\n"
                "5D Chess Rules:\n"
                "1. Movement axes:\n"
                "   - x-axis: Files (a-h)\n"
                "   - y-axis: Ranks (1-8)\n"
                "   - turn axis: Time progression (left to right)\n"
                "   - timeline axis: Parallel universes (vertical)\n\n"
                "2. Piece Movement Rules:\n"
                "   - Rook: Any distance along exactly one axis\n"
                "   - Bishop: Any distance along exactly two axes equally\n"
                "   - Queen: Any distance along any number of axes equally\n"
                "   - King: One space along any number of axes\n"
                "   - Knight: Two spaces along one axis, then one space along another\n"
                "   - Pawn: One space forward (y-axis or timeline axis), captures diagonally\n\n"
                "3. Move Format Requirements:\n"
                "   - Regular moves: EXACTLY 4 characters in UCI format (e.g., 'e2e4', 'b8c6')\n"
                "   - Time travel moves: 'time_travel e2e4 t0' (where t0 is source timeline)\n"
                "   - ONLY use lowercase a-h and numbers 1-8\n"
                "   - NO piece symbols (N, B, R, Q, K) allowed\n\n"
                "4. Timeline Rules:\n"
                "   - New timeline created when moving to an unplayable board\n"
                "   - Can only move on boards where it's your turn\n"
                "   - Must make moves until present line shifts to opponent's turn\n"
                "   - Maximum of 5 active timelines allowed\n\n"
                "IMPORTANT:\n"
                "- Consider movement through ALL dimensions (space, time, and timelines)\n"
                "- Look for tactical opportunities across timelines\n"
                "- Ensure moves are legal in the multiverse context\n"
                "- MOVES MUST BE EXACTLY 4 CHARACTERS for regular moves\n"
                "Your task is to output ONLY your chosen move in exact UCI format, with no additional text."
            )
            
            # Get response from agent
            response = self.agent.run(prompt)
            
            # Clean up response and extract move
            move = response.strip().split()[0]  # Take first word as move
            
            # Validate move format and rules
            is_valid, reason = self.is_valid_5d_move(board, move, color, timeline_id)
            
            if not is_valid:
                retry_prompt = (
                    f"Invalid move ({reason}). Please try again with a valid move following 5D chess rules.\n"
                    "Remember:\n"
                    "- Use exact UCI format (e.g., 'e2e4', 'b8c6')\n"
                    "- Ensure the move follows the piece's movement rules\n"
                    "- Consider timeline and turn constraints\n"
                    "Try again:"
                )
                move = self.agent.run(retry_prompt).strip().split()[0]
                is_valid, reason = self.is_valid_5d_move(board, move, color, timeline_id)
            
            # If still invalid, try emergency move
            if not is_valid:
                # Try to find a valid pawn move first
                for rank in range(8):
                    for file in range(8):
                        piece = board[rank][file]
                        if piece and ((color == "White" and piece.isupper()) or 
                                    (color == "Black" and piece.islower())):
                            from_square = f"{chr(97+file)}{8-rank}"
                            for to_rank in range(8):
                                for to_file in range(8):
                                    to_square = f"{chr(97+to_file)}{8-to_rank}"
                                    test_move = from_square + to_square
                                    is_valid, _ = self.is_valid_5d_move(board, test_move, color, timeline_id)
                                    if is_valid:
                                        move = test_move
                                        break
                            if is_valid:
                                break
                    if is_valid:
                        break
            
            # Generate reasoning based on the move
            from_square = move[0:2]
            to_square = move[2:4]
            piece_type = None
            for rank in range(8):
                for file in range(8):
                    if chr(97+file) + str(8-rank) == from_square:
                        piece_type = board[rank][file]
                        break
                if piece_type:
                    break
            
            piece_names = {
                'P': 'pawn', 'p': 'pawn',
                'N': 'knight', 'n': 'knight',
                'B': 'bishop', 'b': 'bishop',
                'R': 'rook', 'r': 'rook',
                'Q': 'queen', 'q': 'queen',
                'K': 'king', 'k': 'king'
            }
            
            reasoning = (
                f"Moving {piece_names.get(piece_type, 'piece')} from {from_square} to {to_square} "
                f"in timeline {timeline_id} following 5D chess rules"
            )
            
            # Record the move
            self.conversation_history.append({
                "chain_of_thought": reasoning,
                "move": move,
                "timestamp": time.strftime("%H:%M:%S")
            })
            
            return move, reasoning
            
        except Exception as e:
            print(f"LangChain agent error: {str(e)}")
            # Generate a simple valid move as last resort
            files = 'abcdefgh'
            ranks = '12345678'
            move = f"{random.choice(files)}{random.choice(ranks)}{random.choice(files)}{random.choice(ranks)}"
            reasoning = "Error recovery move"
            return move, reasoning

# =============================================================================
# High-Level: Enhanced Themed Chess Renderer with Smooth Animations
# =============================================================================

class ChessRenderer(QOpenGLWidget):
    def __init__(self, game_state, parent=None):
        super().__init__(parent)
        self.game_state = game_state
        self.current_timeline = 0
        self.animation_items = []  # List of active animation items
        self.last_move_highlight = None
        self.potential_moves = []
        self.thinking_squares = set()
        self.timeline_animations = {}  # Animations for timeline transitions
        
        # Enhanced animation settings for 5D chess
        self.ANIMATION_SPEED = 0.02  # Speed of animations
        self.RIPPLE_DURATION = 1.5   # Duration of ripple effects
        self.GLOW_INTENSITY = 0.8    # Maximum glow intensity
        self.TURN_AXIS_SPACING = 150  # Horizontal spacing for turn progression
        self.TIMELINE_AXIS_SPACING = 120  # Vertical spacing for timeline branches
        
        # Official 5D chess colors
        self.LIGHT_SQUARE = QColor(240, 217, 181)
        self.DARK_SQUARE = QColor(181, 136, 99)
        self.BORDER_COLOR = QColor(101, 67, 33)
        self.HIGHLIGHT_COLOR = QColor(255, 255, 0, 80)
        self.TIME_TRAVEL_COLOR = QColor(0, 255, 255)
        self.TURN_AXIS_COLOR = QColor(255, 128, 0)  # Orange for turn axis
        self.TIMELINE_AXIS_COLOR = QColor(0, 255, 255)  # Cyan for timeline axis
        
        # Piece colors and effects
        self.WHITE_PIECE_COLOR = QColor(255, 255, 255)
        self.BLACK_PIECE_COLOR = QColor(0, 0, 0)
        self.PIECE_SHADOW_COLOR = QColor(0, 0, 0, 100)
        self.PIECE_GLOW_COLOR = QColor(255, 215, 0, 50)  # Golden glow
        
        self.start_animation_timer()
        self.setMinimumSize(800, 600)

    def start_animation_timer(self):
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self.update_animations)
        self.anim_timer.start(16)  # ~60 FPS for smooth animations

    def update_animations(self):
        finished = []
        for item in self.animation_items:
            item['progress'] += self.ANIMATION_SPEED
            
            # Update special effects based on animation type
            if item['type'] == 'time_travel':
                # Create multiple ripple waves
                item['ripples'] = [
                    (p, min(1.0, (item['progress'] - p * 0.2) * 2))
                    for p in [0, 0.2, 0.4, 0.6]
                    if item['progress'] > p
                ]
            
            if item['progress'] >= 1.0:
                finished.append(item)
        
        for item in finished:
            self.animation_items.remove(item)
        
        self.update()

    def set_potential_moves(self, moves):
        """Set squares and arrows to show AI's thinking process."""
        self.potential_moves = moves
        self.update()

    def set_thinking_squares(self, squares):
        """Set squares that the AI is currently considering."""
        self.thinking_squares = set(squares)
        self.update()

    def clear_thinking_visualization(self):
        """Clear all thinking indicators."""
        self.potential_moves = []
        self.thinking_squares = set()
        self.update()

    def draw_arrow(self, painter, from_square, to_square):
        """Draw an arrow between two squares."""
        board_size = min(self.width(), self.height())
        square_size = board_size / 8
        
        # Calculate center points of squares
        from_x = (from_square[1] + 0.5) * square_size
        from_y = (7 - from_square[0] + 0.5) * square_size
        to_x = (to_square[1] + 0.5) * square_size
        to_y = (7 - to_square[0] + 0.5) * square_size
        
        # Draw arrow line
        pen = QPen(QColor(0, 255, 0, 150))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(from_x, from_y, to_x, to_y)
        
        # Draw arrow head
        angle = math.atan2(to_y - from_y, to_x - from_x)
        arrow_size = square_size * 0.2
        
        # Create arrow head polygon
        arrow_head = QPolygonF([
            QPointF(to_x, to_y),
            QPointF(to_x - arrow_size * math.cos(angle - math.pi/6),
                   to_y - arrow_size * math.sin(angle - math.pi/6)),
            QPointF(to_x - arrow_size * math.cos(angle + math.pi/6),
                   to_y - arrow_size * math.sin(angle + math.pi/6))
        ])
        
        # Fill arrow head
        painter.setBrush(QColor(0, 255, 0, 150))
        painter.drawPolygon(arrow_head)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Draw all timelines
        for timeline_id, board in self.game_state.timelines.items():
            position = self.game_state.timeline_positions[timeline_id]
            
            # Save the current transform
            painter.save()
            
            # Apply timeline position and rotation
            painter.translate(position['x'] + 200, position['y'] + 100)  # Base offset
            painter.rotate(position['angle'])
            
            # Draw the board for this timeline
            self.draw_timeline_board(painter, timeline_id)
            
            # Restore the transform
            painter.restore()
        
        # Draw timeline connection lines
        self.draw_timeline_connections(painter)
        
        # Draw active animations
        self.draw_animations(painter)

    def draw_timeline_connections(self, painter):
        """Draw enhanced timeline connections showing both turn and timeline axes."""
        # Draw turn axis (horizontal progression)
        for timeline_id in self.game_state.timelines:
            pos = self.game_state.timeline_positions[timeline_id]
            turn_number = len([m for m in self.game_state.history if m[0] == timeline_id])
            
            # Draw turn axis line
            start_x = pos['x'] + 200
            start_y = pos['y'] + 100
            end_x = start_x + self.TURN_AXIS_SPACING * turn_number
            
            turn_gradient = QLinearGradient(start_x, start_y, end_x, start_y)
            turn_gradient.setColorAt(0, self.TURN_AXIS_COLOR)
            turn_gradient.setColorAt(1, self.TURN_AXIS_COLOR.lighter(150))
            
            pen = QPen(QBrush(turn_gradient), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(start_x, start_y, end_x, start_y)
        
        # Draw timeline axis connections (vertical branches)
        for timeline_id in self.game_state.timelines:
            # Find parent timeline from history
            for hist_timeline, move, _ in self.game_state.history:
                if "time_travel" in str(move) and move.split()[-1][1:] == str(hist_timeline):
                    start_pos = self.game_state.timeline_positions[hist_timeline]
                    end_pos = self.game_state.timeline_positions[timeline_id]
                    
                    # Calculate actual screen coordinates with timeline axis spacing
                    start_x = start_pos['x'] + 300
                    start_y = start_pos['y'] + 200
                    end_x = end_pos['x'] + 300
                    end_y = end_pos['y'] + self.TIMELINE_AXIS_SPACING
                    
                    # Create timeline branch gradient
                    timeline_gradient = QLinearGradient(start_x, start_y, end_x, end_y)
                    timeline_gradient.setColorAt(0, self.TIMELINE_AXIS_COLOR)
                    timeline_gradient.setColorAt(1, self.TIMELINE_AXIS_COLOR.lighter(150))
                    
                    # Draw timeline connection with glow effect
                    glow_pen = QPen(self.TIMELINE_AXIS_COLOR.lighter(200), 6)
                    painter.setPen(glow_pen)
                    painter.drawLine(start_x, start_y, end_x, end_y)
                    
                    main_pen = QPen(QBrush(timeline_gradient), 2)
                    painter.setPen(main_pen)
                    painter.drawLine(start_x, start_y, end_x, end_y)
                    
                    # Add arrow indicating timeline branch direction
                    self.draw_timeline_arrow(painter, (end_x, end_y), -90)  # Point downward

    def draw_timeline_arrow(self, painter, pos, angle):
        """Draw an arrow indicating timeline direction."""
        arrow_size = 15
        painter.save()
        painter.translate(pos[0], pos[1])
        painter.rotate(angle)
        
        arrow_path = QPainterPath()
        arrow_path.moveTo(0, 0)
        arrow_path.lineTo(-arrow_size/2, -arrow_size)
        arrow_path.lineTo(arrow_size/2, -arrow_size)
        arrow_path.lineTo(0, 0)
        
        painter.setBrush(self.TIMELINE_AXIS_COLOR)
        painter.setPen(Qt.NoPen)
        painter.drawPath(arrow_path)
        painter.restore()

    def draw_timeline_board(self, painter, timeline_id):
        """Draw a single timeline's chess board."""
        board_size = 400  # Fixed size for each board
        square_size = board_size / 8
        
        # Draw outer border with wood texture effect
        border_size = square_size * 0.5
        outer_rect = QRectF(-border_size, -border_size, 
                           board_size + 2*border_size, board_size + 2*border_size)
        
        # Create wood texture gradient for border
        wood_gradient = QLinearGradient(outer_rect.topLeft(), outer_rect.bottomRight())
        wood_gradient.setColorAt(0, QColor(139, 69, 19))
        wood_gradient.setColorAt(0.5, QColor(160, 82, 45))
        wood_gradient.setColorAt(1, QColor(139, 69, 19))
        painter.fillRect(outer_rect, wood_gradient)
        
        # Draw squares with enhanced gradients
        for row in range(8):
            for col in range(8):
                x = col * square_size
                y = row * square_size
                rect = QRectF(x, y, square_size, square_size)
                
                # Create gradient based on square color
                if (row + col) % 2 == 0:
                    grad = QLinearGradient(x, y, x + square_size, y + square_size)
                    grad.setColorAt(0, self.LIGHT_SQUARE.lighter(110))
                    grad.setColorAt(0.5, self.LIGHT_SQUARE)
                    grad.setColorAt(1, self.LIGHT_SQUARE.darker(110))
                else:
                    grad = QLinearGradient(x, y, x + square_size, y + square_size)
                    grad.setColorAt(0, self.DARK_SQUARE.lighter(110))
                    grad.setColorAt(0.5, self.DARK_SQUARE)
                    grad.setColorAt(1, self.DARK_SQUARE.darker(110))
                
                # Draw square with gradient
                painter.fillRect(rect, grad)
                
                # Add subtle inner shadow
                pen = QPen(QColor(0, 0, 0, 30))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawRect(rect)
                
                # Add highlight for last move
                if self.last_move_highlight and (row, col) in self.last_move_highlight:
                    highlight_rect = rect.adjusted(2, 2, -2, -2)
                    painter.fillRect(highlight_rect, self.HIGHLIGHT_COLOR)
        
        # Draw the pieces on this timeline's board
        board = self.game_state.get_board(timeline_id)
        if board:
            self.draw_pieces(painter, board, board_size)
        
        # Draw timeline label
        painter.setPen(QColor(255, 255, 255))
        font = QFont("Arial", 12)
        painter.setFont(font)
        label_rect = QRectF(-border_size, -border_size - 25, board_size + 2*border_size, 20)
        painter.drawText(label_rect, Qt.AlignCenter, f"Timeline {timeline_id}")

    def draw_pieces(self, painter, board, board_size):
        """Draw enhanced chess pieces with realistic designs and effects."""
        square_size = board_size / 8
        
        # Set up the font for pieces - use system fonts that are guaranteed to exist
        if sys.platform == "darwin":  # macOS
            font = QFont("Arial Unicode MS", int(square_size * 0.75))
        else:  # Windows/Linux
            font = QFont("DejaVu Sans", int(square_size * 0.75))
        
        font.setStyleStrategy(QFont.PreferAntialias)
        painter.setFont(font)
        
        # Enhanced Unicode chess pieces with better symbols
        piece_map = {
            'K': '♔', 'Q': '♕', 'R': '♖', 'B': '♗', 'N': '♘', 'P': '♙',
            'k': '♚', 'q': '♛', 'r': '♜', 'b': '♝', 'n': '♞', 'p': '♟'
        }

        # Fallback ASCII pieces if Unicode rendering fails
        ascii_map = {
            'K': 'K', 'Q': 'Q', 'R': 'R', 'B': 'B', 'N': 'N', 'P': 'P',
            'k': 'k', 'q': 'q', 'r': 'r', 'b': 'b', 'n': 'n', 'p': 'p'
        }
        
        for row in range(8):
            for col in range(8):
                piece = board[row][col]
                if piece:
                    x = col * square_size
                    y = row * square_size
                    rect = QRectF(x, y, square_size, square_size)
                    
                    # Create piece gradient based on color
                    if piece.isupper():  # White pieces
                        piece_gradient = QLinearGradient(
                            rect.topLeft(), rect.bottomRight())
                        piece_gradient.setColorAt(0.0, QColor(255, 255, 255))
                        piece_gradient.setColorAt(0.5, QColor(240, 240, 240))
                        piece_gradient.setColorAt(1.0, QColor(220, 220, 220))
                        piece_color = piece_gradient
                        shadow_color = QColor(0, 0, 0, 80)
                        highlight_color = QColor(255, 255, 255, 120)
                    else:  # Black pieces
                        piece_gradient = QLinearGradient(
                            rect.topLeft(), rect.bottomRight())
                        piece_gradient.setColorAt(0.0, QColor(40, 40, 40))
                        piece_gradient.setColorAt(0.5, QColor(20, 20, 20))
                        piece_gradient.setColorAt(1.0, QColor(0, 0, 0))
                        piece_color = piece_gradient
                        shadow_color = QColor(0, 0, 0, 120)
                        highlight_color = QColor(100, 100, 100, 80)
                    
                    # Draw multiple shadows for depth effect
                    for offset in [3, 2]:
                        shadow_rect = rect.translated(offset, offset)
                        painter.setPen(QPen(shadow_color, 2))
                        try:
                            painter.drawText(shadow_rect, Qt.AlignCenter, 
                                           piece_map.get(piece, piece))
                        except:
                            # Fallback to ASCII if Unicode fails
                            painter.drawText(shadow_rect, Qt.AlignCenter, 
                                           ascii_map.get(piece, piece))
                    
                    # Draw piece base with gradient
                    painter.setPen(QPen(piece_color, 2))
                    painter.setBrush(piece_color)
                    
                    # Add glow effect for active pieces
                    if (row, col) in self.thinking_squares:
                        glow_rect = rect.adjusted(-4, -4, 4, 4)
                        glow_gradient = QLinearGradient(
                            glow_rect.topLeft(), glow_rect.bottomRight())
                        glow_gradient.setColorAt(0.0, QColor(255, 215, 0, 100))
                        glow_gradient.setColorAt(1.0, QColor(255, 215, 0, 0))
                        painter.setPen(QPen(QColor(255, 215, 0, 80), 3))
                        painter.setBrush(glow_gradient)
                        painter.drawEllipse(glow_rect)
                    
                    # Draw the main piece
                    painter.setPen(QPen(piece_color, 2))
                    try:
                        painter.drawText(rect, Qt.AlignCenter, piece_map.get(piece, piece))
                    except:
                        # Fallback to ASCII if Unicode fails
                        painter.drawText(rect, Qt.AlignCenter, ascii_map.get(piece, piece))
                    
                    # Add highlight for dimensionality
                    highlight_rect = rect.adjusted(2, 2, -2, -2)
                    painter.setPen(QPen(highlight_color, 1))
                    try:
                        painter.drawText(highlight_rect, Qt.AlignCenter, 
                                       piece_map.get(piece, piece))
                    except:
                        # Fallback to ASCII if Unicode fails
                        painter.drawText(highlight_rect, Qt.AlignCenter, 
                                       ascii_map.get(piece, piece))
                    
                    # Add extra visual effects for special states
                    if self.last_move_highlight and (row, col) in self.last_move_highlight:
                        # Add subtle pulse effect for last moved piece
                        pulse_rect = rect.adjusted(-2, -2, 2, 2)
                        pulse_color = QColor(255, 255, 0, 40)
                        painter.setPen(QPen(pulse_color, 2))
                        painter.setBrush(Qt.NoBrush)
                        painter.drawEllipse(pulse_rect)

    def draw_animations(self, painter):
        """Draw enhanced animations with turn and timeline axis effects."""
        for item in self.animation_items:
            if item['type'] == 'time_travel':
                # Draw turn axis effect (horizontal)
                turn_start = (item['start'][0], item['start'][1])
                turn_end = (item['start'][0] + self.TURN_AXIS_SPACING, item['start'][1])
                
                turn_gradient = QLinearGradient(turn_start[0], turn_start[1],
                                              turn_end[0], turn_end[1])
                turn_gradient.setColorAt(0, self.TURN_AXIS_COLOR)
                turn_gradient.setColorAt(1, self.TURN_AXIS_COLOR.lighter(150))
                
                turn_pen = QPen(QBrush(turn_gradient), 3)
                painter.setPen(turn_pen)
                painter.drawLine(turn_start[0], turn_start[1],
                               turn_start[0] + self.TURN_AXIS_SPACING * item['progress'],
                               turn_start[1])
                
                # Draw timeline axis effect (vertical)
                timeline_gradient = QLinearGradient(item['start'][0], item['start'][1],
                                                  item['end'][0], item['end'][1])
                timeline_gradient.setColorAt(0, self.TIMELINE_AXIS_COLOR)
                timeline_gradient.setColorAt(1, self.TIMELINE_AXIS_COLOR.lighter(150))
                
                timeline_pen = QPen(QBrush(timeline_gradient), 3)
                painter.setPen(timeline_pen)
                painter.drawLine(item['start'][0], item['start'][1],
                               item['end'][0],
                               item['start'][1] + (item['end'][1] - item['start'][1]) * item['progress'])
                
                # Draw ripple effects at branch points
                for pos in [item['start'], (item['end'][0], item['start'][1] + 
                          (item['end'][1] - item['start'][1]) * item['progress'])]:
                    for _, ripple_progress in item['ripples']:
                        if 0 <= ripple_progress <= 1:
                            radius = ripple_progress * 40
                            opacity = int(255 * (1 - ripple_progress))
                            
                            # Timeline axis ripple
                            timeline_color = QColor(self.TIMELINE_AXIS_COLOR)
                            timeline_color.setAlpha(opacity)
                            painter.setPen(QPen(timeline_color, 2))
                            painter.drawEllipse(QPointF(pos[0], pos[1]), radius, radius)
                            
                            # Turn axis ripple
                            turn_color = QColor(self.TURN_AXIS_COLOR)
                            turn_color.setAlpha(opacity)
                            painter.setPen(QPen(turn_color, 2))
                            painter.drawEllipse(QPointF(pos[0], pos[1]), radius * 0.8, radius * 0.8)

    def add_time_travel_animation(self, from_timeline, to_timeline):
        """Add enhanced time travel animation with turn and timeline axis effects."""
        start_pos = self.game_state.timeline_positions[from_timeline]
        end_pos = self.game_state.timeline_positions[to_timeline]
        
        # Calculate board centers with axis spacing
        board_size = 400
        start_center = (
            start_pos['x'] + board_size/2 + 200,
            start_pos['y'] + board_size/2 + 100
        )
        end_center = (
            end_pos['x'] + board_size/2 + 200,
            end_pos['y'] + board_size/2 + self.TIMELINE_AXIS_SPACING
        )
        
        # Create animation with both turn and timeline components
        anim = {
            'type': 'time_travel',
            'start': start_center,
            'end': end_center,
            'progress': 0,
            'ripples': [],
            'glow_intensity': 0,
            'turn_progress': 0,  # For turn axis animation
            'timeline_progress': 0  # For timeline axis animation
        }
        self.animation_items.append(anim)

    def highlight_last_move(self, squares):
        """Highlight the squares involved in the last move."""
        self.last_move_highlight = squares

# =============================================================================
# High-Level: Chat Log Widget Displaying Formatted Chain-of-Thought and Moves
# =============================================================================

class ChatLogWidget(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setStyleSheet("""
            QTextEdit {
                background-color: #2b2b2b;
                color: #ffffff;
                font-family: 'Arial';
                font-size: 12pt;
                border: none;
                padding: 10px;
            }
        """)
        self.append_system_message("Game Started", "Welcome to 5D Chess with Multiverse Time Travel!")
    
    def append_system_message(self, title, message):
        """Add a system message to the chat log."""
        html = f"""
        <div style='margin: 10px 0; padding: 5px; background-color: #363636; border-radius: 5px;'>
            <span style='color: #66d9ef; font-weight: bold;'>{title}</span>
            <div style='margin-left: 10px; color: #e6e6e6;'>{message}</div>
        </div>
        """
        self.append(html)
        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())
    
    def append_message(self, player_name, move, chain_of_thought):
        """Append a nicely formatted HTML message to the chat log."""
        timestamp = time.strftime("%H:%M:%S")
        
        # Clean up the chain of thought to remove any technical output
        cleaned_cot = []
        for line in chain_of_thought.split('\n'):
            # Skip lines that are just tool usage or technical output
            if any(skip in line.lower() for skip in ['action:', 'action input:', 'observation:', 'entering new']):
                continue
            cleaned_line = line.strip()
            if cleaned_line:
                cleaned_cot.append(cleaned_line)
        
        # Format the cleaned chain of thought with bullet points
        if len(cleaned_cot) > 1:
            cot_formatted = "<ul style='margin: 5px 0;'>"
            for line in cleaned_cot:
                if line.strip():
                    cot_formatted += f"<li>{line.strip()}</li>"
            cot_formatted += "</ul>"
        else:
            cot_formatted = f"<p style='margin: 5px 0;'>{cleaned_cot[0] if cleaned_cot else 'Making move.'}</p>"
        
        html_message = f"""
        <div style='margin: 10px 0; padding: 10px; background-color: #363636; border-radius: 5px;'>
            <div style='display: flex; justify-content: space-between; margin-bottom: 5px;'>
                <span style='color: #a6e22e; font-weight: bold;'>{player_name}</span>
                <span style='color: #75715e;'>{timestamp}</span>
            </div>
            <div style='margin: 5px 0;'>
                <span style='color: #66d9ef; font-weight: bold;'>Move: </span>
                <span style='color: #e6e6e6;'>{move}</span>
            </div>
            <div style='margin: 5px 0;'>
                <span style='color: #f92672; font-weight: bold;'>Reasoning:</span>
                <div style='margin-left: 15px; color: #e6e6e6;'>{cot_formatted}</div>
            </div>
        </div>
        """
        self.append(html_message)
        # Scroll to bottom
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

# =============================================================================
# High-Level: Main Window Combining Renderer and Chat Log with a Game Loop
# =============================================================================

class MainWindow(QMainWindow):
    def __init__(self, game_state, llm1, llm2):
        super().__init__()
        self.game_state = game_state
        self.renderer = ChessRenderer(game_state)
        
        # Update LLM players with renderer reference
        llm1.renderer = self.renderer
        llm2.renderer = self.renderer
        
        self.llm1 = llm1
        self.llm2 = llm2
        self.current_player = self.llm1
        self.current_timeline = 0
        self.move_in_progress = False  # Add lock to prevent multiple moves
        self.init_ui()
        self.move_timer = QTimer(self)
        self.move_timer.timeout.connect(self.game_loop)
        self.move_timer.start(6000)

    def init_ui(self):
        self.setWindowTitle("Themed 5D Chess with Multiverse Time Travel")
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout(central_widget)
        
        # Create the enhanced chess board renderer
        layout.addWidget(self.renderer, stretch=3)
        
        # Create the chat log to show LLM outputs and chain-of-thought
        self.chat_log = ChatLogWidget()
        layout.addWidget(self.chat_log, stretch=1)
    
    def game_loop(self):
        """Run the game loop in a separate thread to avoid blocking the GUI."""
        if not self.move_in_progress:
            threading.Thread(target=self.process_move, daemon=True).start()
    
    def process_move(self):
        """Process the next move in the game."""
        if self.move_in_progress:
            return
        
        try:
            self.move_in_progress = True
            
            # Get the move from the current player
            move, cot = self.current_player.get_move(self.game_state)
            
            # Clean up the move format
            move = move.strip()
            
            # Update the chat log
            self.chat_log.append_message(
                self.current_player.name,
                move,
                cot
            )
            
            # Update the game state
            if "time_travel" in move:
                # Parse time travel move format: "time_travel e2e4 t0"
                parts = move.split()
                actual_move = parts[1].strip()
                source_timeline = int(parts[2][1:])  # Remove 't' prefix
                
                # Make the move and get the new timeline
                new_timeline = self.game_state.make_move(source_timeline, move, cot)
                self.current_timeline = new_timeline
                
                # Add time travel animation
                self.renderer.add_time_travel_animation(source_timeline, new_timeline)
                
                # Update highlight for the actual move
                if len(actual_move) >= 4:
                    from_file = ord(actual_move[0]) - ord('a')
                    from_rank = 8 - int(actual_move[1])
                    to_file = ord(actual_move[2]) - ord('a')
                    to_rank = 8 - int(actual_move[3])
                    
                    self.renderer.last_move_highlight = {
                        (from_rank, from_file),
                        (to_rank, to_file)
                    }
            else:
                # Regular move (e.g., "e2e4")
                move = move.replace(" ", "")
                if len(move) >= 4:
                    # Make the move
                    self.game_state.make_move(self.current_timeline, move, cot)
                    
                    # Update highlight
                    from_file = ord(move[0]) - ord('a')
                    from_rank = 8 - int(move[1])
                    to_file = ord(move[2]) - ord('a')
                    to_rank = 8 - int(move[3])
                    
                    self.renderer.last_move_highlight = {
                        (from_rank, from_file),
                        (to_rank, to_file)
                    }
                else:
                    raise ValueError(f"Invalid move format: {move}")
            
            # Force a redraw
            self.renderer.update()
            
            # Switch players
            self.current_player = self.llm2 if self.current_player == self.llm1 else self.llm1
            
        except Exception as e:
            print(f"Error processing move: {str(e)}")
            self.chat_log.append_system_message("Error", f"An error occurred: {str(e)}")
        finally:
            self.move_in_progress = False

# =============================================================================
# Low-Level: Application Entry Point
# =============================================================================

def main():
    game_state = GameState()
    # Use environment variables for API keys
    llm1 = LLMPlayer("LLM_Player_1", os.getenv("OPENAI_API_KEY"))
    llm2 = LLMPlayer("LLM_Player_2", os.getenv("OPENAI_API_KEY"))
    
    app = QApplication(sys.argv)
    main_window = MainWindow(game_state, llm1, llm2)
    main_window.resize(1000, 700)
    main_window.show()
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()