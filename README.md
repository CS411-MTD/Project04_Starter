# UIC CS 411: Artificial Intelligence
## Project 04: Adversarial Search — Connect Four Starter Code

Welcome to **Project 04: Adversarial Search**. In this project, you are provided with a complete, executable starter program for the two-player game **Connect Four**, featuring an AI agent that makes decisions using adversarial search techniques (Minimax and Alpha-Beta Pruning) and heuristic state evaluation.

---

### Objective

Your goal in this assignment is to:
1. Run and interact with the game in different modes (Human vs AI, AI vs AI, Human vs Human).
2. Carefully inspect, read, and understand the provided starter code.
3. Investigate the game's behavior, identify any logical, algorithmic, and rule-based flaws present in the implementation, and fix them.
4. Verify that your corrected implementation plays legally, evaluates positions soundly, and applies Minimax and Alpha-Beta Pruning correctly and efficiently.
5. Prepare to demonstrate and explain your corrected program and adversarial search concepts during your 1-on-1 project interview.

---

### Getting Started

#### Prerequisites
- Python 3.8 or higher.
- No external third-party libraries are required; the project uses Python standard libraries.

#### Running the Game
To launch the interactive game menu:
```bash
python3 connect4.py
```

You can also run directly with command-line flags:
```bash
# Human vs AI using Alpha-Beta Pruning (Search Depth 4)
python3 connect4.py --mode human-vs-ai --algo alphabeta --depth 4

# Human vs AI using Pure Minimax
python3 connect4.py --mode human-vs-ai --algo minimax --depth 3

# AI vs AI self-play demonstration
python3 connect4.py --mode ai-vs-ai --algo alphabeta --depth 4

# Human vs Human two-player local mode
python3 connect4.py --mode human-vs-human
```

---

### Project Structure

```text
Project04_Starter/
├── connect4.py          # Main Connect Four game engine and AI search algorithms
├── README.md            # Project overview and run instructions
└── requirements.txt     # Environment specifications
```

---

### Assessment Information

- **Assessment Format:** Individual 1-on-1 interview with the course instructor/TA.
- **Demonstration:** You will run your corrected program live, demonstrate game mechanics and AI decision-making across test scenarios, and trace the behavior of Minimax, Alpha-Beta pruning, and evaluation heuristics.
- **Code Defense:** You will explain the problems you discovered, why they occurred, how you fixed them, and how your changes ensure algorithmic correctness and search efficiency.

