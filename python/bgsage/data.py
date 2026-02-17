"""
Data loading for GNUbg benchmark and training files.

Parses .bm benchmark files into structured data that the C++ engine can consume.
"""

import sys
import os

# Add the build directory to find the compiled C++ module
_BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'build')
if os.path.isdir(_BUILD_DIR):
    # On Windows, we need to add DLL directory for MinGW runtime
    if sys.platform == 'win32':
        os.add_dll_directory(os.path.abspath(_BUILD_DIR))
    sys.path.insert(0, os.path.abspath(_BUILD_DIR))

import bgbot_cpp


def board_from_gnubg_position_string(pos_str: str) -> list[int]:
    """
    Decode a 20-character GNUbg position string into a 26-element board list.

    The encoding is GNUbg's internal "nn" format. Each pair of characters encodes
    8 bits. Bits are scanned left to right: a 1 bit means "add one more checker
    to the current point", a 0 bit means "move to the next point".

    Returns a list of 26 ints:
    - [0]: player 2 checkers on bar (>= 0)
    - [1-24]: board points (positive = player 1, negative = player 2)
    - [25]: player 1 checkers on bar (>= 0)
    """
    if len(pos_str) != 20:
        raise ValueError(f'Invalid gnubg position string (length {len(pos_str)})')

    # Decode 10 bytes from pairs of characters
    key = [0] * 10
    for i in range(10):
        key[i] = ((ord(pos_str[2 * i]) - ord('A')) << 4) + (ord(pos_str[2 * i + 1]) - ord('A'))

    checkers = [0] * 26

    # i tracks which player (0 = player 2 / black, 1 = player 1 / white)
    # j tracks which point (0-23 for points, 24 for bar)
    i = j = 0

    for ind in range(10):
        cur = key[ind]
        for _ in range(8):
            if cur & 0x1:
                if j < 24:
                    if i == 0:
                        checkers[24 - j] -= 1  # player 2 checker
                    else:
                        checkers[j + 1] += 1   # player 1 checker
                else:
                    if i == 0:
                        checkers[0] += 1   # player 2 on bar
                    else:
                        checkers[25] += 1  # player 1 on bar
            else:
                j += 1
                if j == 25:
                    i += 1
                    j = 0
            cur >>= 1

    return checkers


def load_benchmark_file(filepath: str, step: int = 1):
    """
    Load a .bm benchmark file and parse all "move" scenarios.

    Args:
        filepath: Path to the .bm file (contact.bm, crashed.bm, or race.bm)
        step: Take every Nth scenario (1 = all, 2 = every other, etc.)

    Returns:
        A bgbot_cpp.ScenarioSet with all scenarios stored in C++.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # Filter to move lines only
    move_lines = [line for line in lines if line.startswith('m ')]

    # Subsample
    move_lines = move_lines[::step]

    ss = bgbot_cpp.ScenarioSet()
    ss.reserve(len(move_lines))

    for line in move_lines:
        bits = line.split()

        # Parse starting position
        start_board = board_from_gnubg_position_string(bits[1])
        die1 = int(bits[2])
        die2 = int(bits[3])

        # Parse ranked result boards and their errors
        ranked_boards = []
        ranked_errors = []

        idx = 4
        while idx < len(bits):
            board = board_from_gnubg_position_string(bits[idx])
            ranked_boards.append(board)

            if idx + 1 < len(bits):
                ranked_errors.append(float(bits[idx + 1]))
            else:
                ranked_errors.append(0.0)

            idx += 2

        ss.add(start_board, die1, die2, ranked_boards, ranked_errors)

    return ss


def load_benchmark_scenarios_by_indices(filepath: str, indices: list):
    """
    Load specific scenarios from a .bm file by their 0-based indices.

    Args:
        filepath: Path to the .bm file
        indices: Sorted list of 0-based scenario indices to load

    Returns:
        A bgbot_cpp.ScenarioSet with only the requested scenarios.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    move_lines = [line for line in lines if line.startswith('m ')]

    indices_set = set(indices)
    ss = bgbot_cpp.ScenarioSet()
    ss.reserve(len(indices))

    for i, line in enumerate(move_lines):
        if i not in indices_set:
            continue

        bits = line.split()
        start_board = board_from_gnubg_position_string(bits[1])
        die1 = int(bits[2])
        die2 = int(bits[3])

        ranked_boards = []
        ranked_errors = []
        idx = 4
        while idx < len(bits):
            board = board_from_gnubg_position_string(bits[idx])
            ranked_boards.append(board)
            if idx + 1 < len(bits):
                ranked_errors.append(float(bits[idx + 1]))
            else:
                ranked_errors.append(0.0)
            idx += 2

        ss.add(start_board, die1, die2, ranked_boards, ranked_errors)

    return ss


def load_gnubg_training_data(filepath: str):
    """
    Load a GNUbg training data file.

    Each line is: <20-char position string> <Pwin> <Pgw> <Pbw> <Pgl> <Pbl>

    The positions are PRE-ROLL, from the perspective of the player on roll.
    For training we need POST-MOVE boards, so we flip the board and probabilities.

    Returns:
        boards: numpy array of int32, shape [N, 26]
        targets: numpy array of float32, shape [N, 5]
    """
    import numpy as np

    boards = []
    targets = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            parts = line.split()
            if len(parts) != 6:
                continue

            pos_str = parts[0]
            if len(pos_str) != 20:
                continue

            try:
                board = board_from_gnubg_position_string(pos_str)
            except (ValueError, IndexError):
                continue

            probs = [float(x) for x in parts[1:6]]

            # The data is from the on-roll player's perspective (pre-roll).
            # For supervised training, we need to flip both the board and the probabilities
            # because the NN evaluates post-move boards from the opponent's perspective.
            #
            # Flip board:
            flipped = [0] * 26
            flipped[0] = board[25]
            flipped[25] = board[0]
            for i in range(1, 25):
                flipped[i] = -board[25 - i]

            # Flip probabilities: [Pwin, Pgw, Pbw, Pgl, Pbl] -> [1-Pwin, Pgl, Pbl, Pgw, Pbw]
            p_win, p_gw, p_bw, p_gl, p_bl = probs
            flipped_probs = [1.0 - p_win, p_gl, p_bl, p_gw, p_bw]

            boards.append(flipped)
            targets.append(flipped_probs)

    boards = np.array(boards, dtype=np.int32)
    targets = np.array(targets, dtype=np.float32)

    return boards, targets
