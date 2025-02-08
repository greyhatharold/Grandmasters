# Grandmaster: Multi-Dimensional Chess AI Visualization

A sophisticated chess visualization platform that lets you witness ChatGPT playing against itself in 2D, 3D, and 5D time travel chess. This project was inspired by Gotham Chess's failure to impress me with their AI demonstrations and the mind-bending game [5D Chess With Multiverse Time Travel](https://store.steampowered.com/app/1349230/5D_Chess_With_Multiverse_Time_Travel/) which I am not profitting from implementing, this is purely for fun.

## Overview

Grandmaster provides three distinct chess experiences, each adding a new dimension of complexity:

1. **2D Chess (Classical)**
   - Traditional chess visualization with LangChain-enabled AI moves
   - Clean GUI interface using Tkinter
   - Real-time move generation using OpenAI's GPT models
   - Detailed chain-of-thought reasoning display

2. **3D Chess**
   - Three-level chess board implementation
   - Vertical movement between levels
   - Enhanced visualization with level-based piece tracking
   - Animated chat log showing AI reasoning
   - Custom 3D move notation

3. **5D Chess with Multiverse Time Travel**
   - Full implementation of official 5D Chess rules
   - Timeline branching and parallel universe exploration
   - Rich visualization of temporal and spatial moves
   - Advanced piece movement across dimensions
   - Timeline management and multiverse navigation
   - Themed rendering with smooth animations

## Features

- **AI vs AI Gameplay**: Watch two instances of ChatGPT play against each other
- **Chain-of-Thought Visualization**: See the AI's reasoning process in real-time
- **Multiple Dimensions**: Experience chess in 2D, 3D, and 5D with time travel
- **Rich Visualizations**: Beautiful UI with themed boards and animated moves
- **Real-time Analysis**: Live commentary and move explanations
- **Timeline Management**: Track and visualize parallel universes in 5D chess

## Requirements

```bash
# Core dependencies
python >= 3.8
openai
langchain
python-chess  # for 2D chess
PySide6      # for 5D chess
tkinter      # included with Python, for 2D/3D chess

# Install required packages
pip install openai langchain python-chess PySide6
```

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Grandmaster.git
   cd Grandmaster
   ```

2. Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### 2D Chess
```bash
python src/2d_chess.py
```
- Launches a traditional chess board with AI vs AI gameplay
- Shows move generation and reasoning in real-time
- Uses python-chess for move validation

### 3D Chess
```bash
python src/3d_chess.py
```
- Displays three chess boards representing different levels
- Allows pieces to move vertically between levels
- Features animated chat log with AI reasoning

### 5D Chess with Multiverse Time Travel
```bash
python src/5d_chess.py
```
- Implements full 5D chess rules with timeline branching
- Visualizes multiple timelines and parallel universes
- Shows temporal and spatial moves with animations
- Includes advanced UI with themed rendering

## File Structure

```
Grandmaster/
├── src/
│   ├── 2d_chess.py    # Traditional chess implementation
│   ├── 3d_chess.py    # Three-level chess visualization
│   ├── 5d_chess.py    # 5D chess with multiverse time travel
├── .env               # Environment variables (API keys)
└── README.md         # This file
```

## Technical Details

### 2D Chess (`2d_chess.py`)
- Uses python-chess for move validation and board state
- Implements LangChain agent for move generation
- Features Tkinter-based GUI with Unicode chess pieces
- Includes console-based board visualization tool

### 3D Chess (`3d_chess.py`)
- Custom 3D chess rules with three vertical levels
- Animated chat log showing AI reasoning
- Level-based move validation and piece tracking
- Enhanced visualization with multiple board views

### 5D Chess (`5d_chess.py`)
- Complete implementation of official 5D chess rules
- Timeline branching and parallel universe management
- Advanced rendering with PySide6 and OpenGL
- Sophisticated move validation across dimensions
- Rich animations and visual effects

## Contributing

Feel free to submit issues, fork the repository, and create pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by [5D Chess With Multiverse Time Travel](https://store.steampowered.com/app/1349230/5D_Chess_With_Multiverse_Time_Travel/)
- Built using OpenAI's GPT models and LangChain
- Special thanks to the chess programming community

## Contact

For questions and support, please open an issue in the GitHub repository. 