#!/usr/bin/env python3
"""
CS 411: Artificial Intelligence
Project 04: Adversarial Search — Connect Four
Starter Code (AI-Assisted Draft)

A Connect Four game engine featuring Minimax and Alpha-Beta Pruning.
"""

import sys
import math
import time
import random
import argparse

# Board Dimensions
ROWS = 6
COLS = 7

# Piece Identifiers
EMPTY = 0
PLAYER_PIECE = 1  # Human (X)
AI_PIECE = 2      # AI (O)

WINDOW_LENGTH = 4


def create_board():
    """Initializes and returns an empty 6x7 game board."""
    return [[EMPTY for _ in range(COLS)] for _ in range(ROWS)]


def print_board(board):
    """
    Renders the game board to the terminal.
    Row 0 is the bottom floor of the grid; row 5 is the top.
    """
    symbols = {EMPTY: '.', PLAYER_PIECE: 'X', AI_PIECE: 'O'}
    print("\n  " + "   ".join(str(c) for c in range(COLS)))
    print("+" + "---+" * COLS)
    for r in range(ROWS - 1, -1, -1):
        row_str = " | ".join(symbols[board[r][c]] for c in range(COLS))
        print(f"| {row_str} |")
        print("+" + "---+" * COLS)
    print("  " + "   ".join(str(c) for c in range(COLS)) + "\n")


def drop_piece(board, row, col, piece):
    """Places a game piece on the board at the designated (row, col)."""
    board[row][col] = piece


def get_next_open_row(board, col):
    """Finds the next available row in the specified column."""
    for r in range(ROWS - 1, -1, -1):
        if board[r][col] == EMPTY:
            return r
    return None


def is_valid_location(board, col):
    """Checks whether it is legal to drop a piece into the specified column."""
    if not (0 <= col < COLS):
        return False
    return board[0][col] == EMPTY


def get_valid_locations(board):
    """Returns a list of all column indices where a piece can legally be dropped."""
    valid_locations = []
    for col in range(COLS):
        if is_valid_location(board, col):
            valid_locations.append(col)
    return valid_locations


def check_win(board, piece):
    """
    Determines whether the specified player has achieved 4-in-a-row
    horizontally, vertically, or along either diagonal.
    """
    # Check horizontal locations for win
    for r in range(ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == piece and 
                board[r][c+1] == piece and 
                board[r][c+2] == piece and 
                board[r][c+3] == piece):
                return True

    # Check vertical locations for win
    for c in range(COLS):
        for r in range(ROWS - 3):
            if (board[r][c] == piece and 
                board[r+1][c] == piece and 
                board[r+2][c] == piece and 
                board[r+3][c] == piece):
                return True

    # Check positively sloped diagonals (/)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            if (board[r][c] == piece and 
                board[r+1][c+1] == piece and 
                board[r+2][c+2] == piece and 
                board[r+3][c+3] == piece):
                return True

    # Check negatively sloped diagonals (\)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            if (board[r][c] == piece and 
                board[r-1][c+1] == piece and 
                board[r-2][c+2] == piece and 
                board[r-3][c+2] == piece):
                return True

    return False


def is_terminal_node(board):
    """Checks if the game has ended due to a win or a full board."""
    return (check_win(board, PLAYER_PIECE) or 
            check_win(board, AI_PIECE) or 
            len(get_valid_locations(board)) == 0)


def evaluate_window(window, piece):
    """
    Heuristic evaluation function for a 4-cell window.
    Assigns numerical utility based on piece counts.
    """
    score = 0
    opp_piece = PLAYER_PIECE if piece == AI_PIECE else AI_PIECE

    if window.count(piece) == 4:
        score += 10000
    elif window.count(piece) == 3 and window.count(EMPTY) == 1:
        score += 50
    elif window.count(piece) == 2 and window.count(EMPTY) == 2:
        score += 10

    if window.count(opp_piece) == 3 and window.count(EMPTY) == 1:
        score += 80

    return score


def score_position(board, piece):
    """
    Calculates the heuristic utility of the full board state for the given player.
    """
    score = 0

    # Center column preference
    center_array = [board[r][COLS // 2] for r in range(ROWS)]
    center_count = center_array.count(piece)
    score -= center_count * 6

    # Horizontal windows
    for r in range(ROWS):
        row_array = [board[r][c] for c in range(COLS)]
        for c in range(COLS - 3):
            window = row_array[c:c + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Vertical windows
    for c in range(COLS):
        col_array = [board[r][c] for r in range(ROWS)]
        for r in range(ROWS - 3):
            window = col_array[r:r + WINDOW_LENGTH]
            score += evaluate_window(window, piece)

    # Positive diagonal windows (/)
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r+i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    # Negative diagonal windows (\)
    for r in range(3, ROWS):
        for c in range(COLS - 3):
            window = [board[r-i][c+i] for i in range(WINDOW_LENGTH)]
            score += evaluate_window(window, piece)

    return score


def minimax(board, depth, maximizing_player, stats=None):
    """
    Standard Minimax adversarial search without pruning.
    Returns:
        tuple: (best_column, best_score)
    """
    if stats is not None:
        stats['nodes'] += 1

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if check_win(board, AI_PIECE):
                return (None, -10000000)
            elif check_win(board, PLAYER_PIECE):
                return (None, 10000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizing_player:
        max_eval = -math.inf
        best_col = valid_locations[0] if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            drop_piece(board, row, col, AI_PIECE)
            _, current_eval = minimax(board, depth - 1, False, stats)
            board[0][col] = EMPTY

            if current_eval > max_eval:
                max_eval = current_eval
                best_col = col

        return best_col, max_eval

    else:
        min_eval = -math.inf
        best_col = valid_locations[0] if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            drop_piece(board, row, col, PLAYER_PIECE)
            _, current_eval = minimax(board, depth - 1, True, stats)
            board[0][col] = EMPTY

            if current_eval > min_eval:
                min_eval = current_eval
                best_col = col

        return best_col, min_eval


def minimax_alphabeta(board, depth, alpha, beta, maximizing_player, stats=None):
    """
    Minimax search enhanced with Alpha-Beta pruning.
    Returns:
        tuple: (best_column, best_score)
    """
    if stats is not None:
        stats['nodes'] += 1

    valid_locations = get_valid_locations(board)
    is_terminal = is_terminal_node(board)

    if depth == 0 or is_terminal:
        if is_terminal:
            if check_win(board, AI_PIECE):
                return (None, -10000000)
            elif check_win(board, PLAYER_PIECE):
                return (None, 10000000)
            else:
                return (None, 0)
        else:
            return (None, score_position(board, AI_PIECE))

    if maximizing_player:
        max_eval = -math.inf
        best_col = valid_locations[0] if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            drop_piece(board, row, col, AI_PIECE)
            _, current_eval = minimax_alphabeta(board, depth - 1, alpha, beta, False, stats)
            board[0][col] = EMPTY

            if current_eval > max_eval:
                max_eval = current_eval
                best_col = col

            alpha = max(alpha, max_eval)
            if alpha <= beta:
                break

        return best_col, max_eval

    else:
        min_eval = -math.inf
        best_col = valid_locations[0] if valid_locations else None
        for col in valid_locations:
            row = get_next_open_row(board, col)
            if row is None:
                continue
            drop_piece(board, row, col, PLAYER_PIECE)
            _, current_eval = minimax_alphabeta(board, depth - 1, alpha, beta, True, stats)
            board[0][col] = EMPTY

            if current_eval > min_eval:
                min_eval = current_eval
                best_col = col

            alpha = min(alpha, min_eval)
            if alpha >= beta:
                break

        return best_col, alpha


def get_ai_move(board, algorithm="alphabeta", depth=4):
    """
    Computes and returns the AI's selected column using the chosen algorithm.
    Also returns a performance dictionary containing node count and elapsed time.
    """
    stats = {'nodes': 0}
    start_time = time.time()
    valid_cols = get_valid_locations(board)

    # Work on a copy of the board during search
    board_copy = [row[:] for row in board]

    if algorithm == "alphabeta":
        col, score = minimax_alphabeta(board_copy, depth, -math.inf, math.inf, True, stats)
    elif algorithm == "minimax":
        col, score = minimax(board_copy, depth, True, stats)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    if (col is None or col not in valid_cols) and valid_cols:
        col = random.choice(valid_cols)

    elapsed_ms = (time.time() - start_time) * 1000
    stats['time_ms'] = elapsed_ms
    stats['score'] = score
    return col, stats


def play_game(mode="human-vs-ai", algorithm="alphabeta", depth=4):
    """Main interactive gameplay loop."""
    board = create_board()
    game_over = False
    turn = random.choice([0, 1])  # 0: Player 1 (X), 1: Player 2 (O)

    print("\n========================================================")
    print("      UIC CS 411: Adversarial Search - Connect Four")
    print(f"  Mode: {mode} | Algorithm: {algorithm} | Search Depth: {depth}")
    print("========================================================")
    print_board(board)

    if mode == "human-vs-ai":
        p1_name = "Human (X)"
        p2_name = f"AI ({algorithm.upper()}, O)"
    elif mode == "ai-vs-ai":
        p1_name = f"AI-1 ({algorithm.upper()}, X)"
        p2_name = f"AI-2 ({algorithm.upper()}, O)"
    else:
        p1_name = "Player 1 (X)"
        p2_name = "Player 2 (O)"

    current_player_name = p1_name if turn == 0 else p2_name
    print(f"Coin toss: {current_player_name} goes first!\n")

    move_count = 0

    while not game_over:
        valid_cols = get_valid_locations(board)
        if not valid_cols:
            print("The board is full! The game is a draw!")
            break

        if turn == 0:
            # Player 1 turn
            if mode == "ai-vs-ai":
                print(f"[Move {move_count + 1}] {p1_name} is calculating move...")
                col, stats = get_ai_move(board, algorithm, depth)
                print(f"{p1_name} selected column {col} "
                      f"(Nodes evaluated: {stats['nodes']}, "
                      f"Time: {stats['time_ms']:.2f}ms, Score: {stats['score']})")
            else:
                # Human player
                prompt = f"[Move {move_count + 1}] {p1_name}, enter column (0-6) or 'q' to quit: "
                try:
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGame exited.")
                    sys.exit(0)

                if user_input.lower() in ('q', 'quit', 'exit'):
                    print("Game aborted by user.")
                    return

                if not user_input.isdigit():
                    print("Invalid input. Please enter an integer between 0 and 6.")
                    continue

                col = int(user_input)
                if col not in valid_cols:
                    print(f"Column {col} is not a legal move! Legal columns: {valid_cols}")
                    continue

            row = get_next_open_row(board, col)
            if row is not None:
                drop_piece(board, row, col, PLAYER_PIECE)
                print_board(board)
                move_count += 1

                if check_win(board, PLAYER_PIECE):
                    print(f"*** {p1_name} WINS! Congratulations! ***\n")
                    game_over = True
                    break
            else:
                print(f"Column {col} is full! Try again.")
                continue

            turn = 1

        else:
            # Player 2 turn
            if mode in ("human-vs-ai", "ai-vs-ai"):
                print(f"[Move {move_count + 1}] {p2_name} is thinking...")
                col, stats = get_ai_move(board, algorithm, depth)
                print(f"{p2_name} selected column {col} "
                      f"(Nodes evaluated: {stats['nodes']}, "
                      f"Time: {stats['time_ms']:.2f}ms, Score: {stats['score']})")
            else:
                # Human player 2
                prompt = f"[Move {move_count + 1}] {p2_name}, enter column (0-6) or 'q' to quit: "
                try:
                    user_input = input(prompt).strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nGame exited.")
                    sys.exit(0)

                if user_input.lower() in ('q', 'quit', 'exit'):
                    print("Game aborted by user.")
                    return

                if not user_input.isdigit():
                    print("Invalid input. Please enter an integer between 0 and 6.")
                    continue

                col = int(user_input)
                if col not in valid_cols:
                    print(f"Column {col} is not a legal move! Legal columns: {valid_cols}")
                    continue

            row = get_next_open_row(board, col)
            if row is not None:
                drop_piece(board, row, col, AI_PIECE)
                print_board(board)
                move_count += 1

                if check_win(board, AI_PIECE):
                    print(f"*** {p2_name} WINS! ***\n")
                    game_over = True
                    break
            else:
                print(f"Column {col} is full! Try again.")
                continue

            turn = 0


def interactive_menu():
    """Presents a command-line menu to select play settings."""
    print("==============================================================")
    print("       UIC CS 411: Artificial Intelligence — Project 04       ")
    print("             Adversarial Search: Connect Four Game            ")
    print("==============================================================")
    print("Select Game Mode:")
    print("  1. Human (X) vs AI (O) [Alpha-Beta Pruning]")
    print("  2. Human (X) vs AI (O) [Pure Minimax]")
    print("  3. AI (X) vs AI (O) [Self-Play Demonstration]")
    print("  4. Human (X) vs Human (O) [Two Players]")
    
    choice = input("\nEnter choice (1-4) [default: 1]: ").strip()
    if choice == '2':
        mode = "human-vs-ai"
        algo = "minimax"
    elif choice == '3':
        mode = "ai-vs-ai"
        algo = "alphabeta"
    elif choice == '4':
        mode = "human-vs-human"
        algo = "alphabeta"
    else:
        mode = "human-vs-ai"
        algo = "alphabeta"

    depth_input = input("Enter AI search depth (1-6) [default: 4]: ").strip()
    try:
        depth = int(depth_input)
        if depth < 1 or depth > 6:
            depth = 4
    except ValueError:
        depth = 4

    play_game(mode=mode, algorithm=algo, depth=depth)


def main():
    parser = argparse.ArgumentParser(
        description="CS 411 Project 04: Connect Four Adversarial Search"
    )
    parser.add_argument(
        "--mode", 
        choices=["human-vs-ai", "ai-vs-ai", "human-vs-human"],
        default=None,
        help="Game execution mode"
    )
    parser.add_argument(
        "--algo", 
        choices=["alphabeta", "minimax"],
        default="alphabeta",
        help="Adversarial search algorithm for AI (alphabeta or minimax)"
    )
    parser.add_argument(
        "--depth", 
        type=int, 
        default=4,
        help="Search depth limit for Minimax / Alpha-Beta (default: 4)"
    )

    args = parser.parse_args()

    if args.mode is None:
        interactive_menu()
    else:
        play_game(mode=args.mode, algorithm=args.algo, depth=args.depth)


if __name__ == "__main__":
    main()
