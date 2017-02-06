"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random


class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
        The heuristic value of the current game state to the specified player.
    """
    return float(heuristic_3(game, player))


def heuristic_1(game, player):
    """Aggressive play along the whole game. Active player will try to choose the most aggressive move.
    Heuristic calculates number of players move vs against 3.5 of value of an opponent’s moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 3.5 * opp_moves)


def heuristic_2(game, player):
    """Aggressive play after the half of the game. Active player will try to choose the most aggressive move.
    Heuristic calculates number of players move vs 3.5 of value of an opponent’s moves.
    In the first half of the game heuristic will calculate number of players move vs 2 of value of an opponent’s moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    cells_left = game.width * game.height - game.move_count

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((game.width * game.height) / 2):
        return float(own_moves - 3 * opp_moves)
    return float(own_moves - 2 * opp_moves)


def heuristic_3(game, player):
    """ Best heuristic among 6 present.
    Aggressive play in the first half of the game. Active player will try to choose the most aggressive move.
    Heuristic calculates number of players move vs 3.5 of value of an opponent’s moves.
    In the second half of the game heuristic will calculate number of players move vs number of an opponent’s moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    cells_left = game.width * game.height - game.move_count

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((game.width * game.height) / 2):
        return float(own_moves - opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_4(game, player):
    """Different level of aggressiveness on three different levels of game.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 0.4):
        return float(own_moves - opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_5(game, player):
    """Similar to H5 but player instead plays less aggressive to the end of the game. (reversed order of aggressiveness)

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 0.4):
        return float(own_moves - 3 * opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - opp_moves)


def heuristic_6(game, player):
    """Similar to H3 but player instead plays less aggressive at the beginning of the game.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 2):
        return float(own_moves - opp_moves)
    return float(own_moves - 2 * opp_moves)


def h_num_moves(game, player):
    """Calculate the number of blank cells for a game state given.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """
    return len(game.get_legal_moves(player))


def h_mine_minus_his(game, player):
    """Calculate the number of blank cells for a game state given.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - opp_moves)


def h_mine_minus_2his(game, player):
    """Calculate the number of blank cells for a game state given.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)


def h_mine_minus_3his(game, player):
    """Calculate the number of blank cells for a game state given.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    Returns
    ----------
    float
        The heuristic value of the current game state.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 3.5 * opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        ----------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring

            max_cell = [float('-inf'), (-1, -1)]  # to track max [score,move]
            if self.method == 'minimax':
                if self.iterative:
                    for depth in range(0, 99):
                        for move in legal_moves:
                            val, _ = self.minimax(game.forecast_move(move), depth, maximizing_player=False)
                            if val > max_cell[0]:
                                max_cell = [val, move]
                else:
                    for move in legal_moves:
                        val, _ = self.minimax(game.forecast_move(move), self.search_depth, maximizing_player=False)
                        if val > max_cell[0]:
                            max_cell = [val, move]
            else:
                if self.iterative:
                    for depth in range(0, 99):
                        for move in legal_moves:
                            val, _ = self.alphabeta(game.forecast_move(move), depth, maximizing_player=False)
                            if val > max_cell[0]:
                                max_cell = [val, move]
                else:
                    for move in legal_moves:
                        val, _ = self.alphabeta(game.forecast_move(move), self.search_depth, maximizing_player=False)
                        if val > max_cell[0]:
                            max_cell = [val, move]
            return max_cell[1]
        except Timeout:
            return max_cell[1]

    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves_available = game.get_legal_moves()  # get list of legal moves for the current player

        if depth == 0 or not moves_available:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            max_cell = [float('-inf'), (-1, -1)]  # to track max [score,move]
            for move in moves_available:
                val, _ = self.minimax(game.forecast_move(move), depth - 1, maximizing_player=False)
                if val > max_cell[0]:
                    max_cell = [val, move]
            return max_cell[0], max_cell[1]
        else:
            min_cell = [float('inf'), (-1, -1)]  # to track min [score,move]
            for move in moves_available:
                val, _ = self.minimax(game.forecast_move(move), depth - 1, maximizing_player=True)
                if val < min_cell[0]:
                    min_cell = [val, move]
            return min_cell[0], min_cell[1]

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        ----------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        moves_available = game.get_legal_moves()  # get list of legal moves for the current player

        if depth == 0 or not moves_available:
            return self.score(game, self), (-1, -1)

        if maximizing_player:
            max_cell = [float('-inf'), (-1, -1)]  # to track max [score,move]
            for move in moves_available:
                val, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, maximizing_player=False)
                if val >= beta:
                    return val, move
                if val > alpha:
                    alpha = val
                    max_cell = [val, move]
            return max_cell[0], max_cell[1]
        else:
            min_cell = [float('inf'), (-1, -1)]  # to track min [score,move]
            for move in moves_available:
                val, _ = self.alphabeta(game.forecast_move(move), depth - 1, alpha, beta, maximizing_player=True)
                if val <= alpha:
                    return val, move
                if val < beta:
                    beta = val
                    min_cell = [val, move]
            return min_cell[0], min_cell[1]
