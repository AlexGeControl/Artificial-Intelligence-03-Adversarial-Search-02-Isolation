"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import numpy as np

class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass

def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # termination:
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # identify opponent:
    opponent = game.get_opponent(player)

    # active moves:
    num_active_moves = len(game.get_legal_moves(player))
    # opponent moves:
    num_opponent_moves = len(game.get_legal_moves(opponent))

    # heuristic score:
    delta = float(num_active_moves - num_opponent_moves)

    score = np.sign(delta) * (delta**2)

    return score

def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # termination:
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    # identify opponent:
    opponent = game.get_opponent(player)

    # active moves:
    num_active_moves = len(game.get_legal_moves(player))
    # opponent moves:
    num_opponent_moves = len(game.get_legal_moves(opponent))

    # heuristic score:
    delta = float(num_active_moves - num_opponent_moves)

    score = delta**3

    return score

def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    # termination:
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    def get_player_center_score(player):
        """Calculate center score for input legal moves
        """
        w, h = game.width / 2., game.height / 2.
        y, x = game.get_player_location(player)

        return (h - y)**2 + (w - x)**2

    # identify opponent:
    opponent = game.get_opponent(player)

    # active center score:
    score_active = get_player_center_score(player)
    # opponent center score:
    score_opponent = get_player_center_score(opponent)

    # heuristic score:
    delta_center_score = float(score_active - score_opponent)
    score_center_score = delta_center_score

    # active moves:
    num_active_moves = len(game.get_legal_moves(player))
    # opponent moves:
    num_opponent_moves = len(game.get_legal_moves(opponent))

    # heuristic score:
    delta_num_moves = float(num_active_moves - num_opponent_moves)
    score_num_moves = np.sign(delta_num_moves) * (delta_num_moves**2)

    score = 0.382 * score_center_score + 0.618 * score_num_moves

    return score

class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            return self.minimax(game, self.search_depth)

        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _maximize(self, active_player, game, depth):
        """ Implement minimization node in adversarial search
        Parameters
        ----------
        active_player : object
            Active player that launches minimax search

        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            input game utility
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # use score function when maximum depth is reached:
        if depth == 0:
            return self.score(game, active_player)

        # termination test:
        termination_utility = game.utility(active_player)
        if termination_utility != 0:
            return termination_utility

        # do maximization:
        max_utility = float("-inf")

        for move in game.get_legal_moves(game.active_player):
            # next game:
            next_game = game.forecast_move(move)
            # next game utility:
            utility = self._minimize(active_player, next_game, depth - 1)
            # update max utility:
            max_utility = max(max_utility, utility)

        return max_utility

    def _minimize(self, active_player, game, depth):
        """ Implement minimization node in adversarial search
        Parameters
        ----------
        active_player : object
            Active player that launches minimax search

        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            input game utility
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # use score function when maximum depth is reached:
        if depth == 0:
            return self.score(game, active_player)

        # termination test:
        termination_utility = game.utility(active_player)
        if termination_utility != 0:
            return termination_utility

        # do minimization:
        min_utility = float("inf")

        for move in game.get_legal_moves(game.active_player):
            # next game:
            next_game = game.forecast_move(move)
            # next game utility:
            utility = self._maximize(active_player, next_game, depth - 1)
            # update min utility:
            min_utility = min(min_utility, utility)

        return min_utility

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # identify active player:
        active_player = game.active_player

        # identify max utility move:
        max_utility = float("-inf")
        max_move = (-1, -1)

        # able to perform search:
        if depth > 0:
            for move in game.get_legal_moves(active_player):
                # next game:
                next_game = game.forecast_move(move)
                # next game utility:
                utility = self._minimize(active_player, next_game, depth - 1)
                # update max utility move:
                if max_utility < utility:
                    max_utility = utility
                    max_move = move

        return max_move

class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = (-1, -1)

        # maximum depth:
        N = game.width * game.height

        try:
            # The try/except block will automatically catch the exception
            # raised when the timer is about to expire.
            for max_depth in range(1, N + 1):
                best_move = self.alphabeta(game, max_depth)
        except SearchTimeout:
            pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def _maximize(self, active_player, game, depth, alpha, beta):
        """ Implement minimization node in adversarial search with alpha-beta pruning
        Parameters
        ----------
        active_player : object
            Active player that launches minimax search

        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            input game utility
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # alpha-beta window test:
        if alpha > beta:
            return float("-inf")

        # use score function when maximum depth is reached:
        if depth == 0:
            return self.score(game, active_player)

        # termination test:
        termination_utility = game.utility(active_player)
        if termination_utility != 0:
            return termination_utility

        # do maximization:
        max_utility = float("-inf")

        for move in game.get_legal_moves(game.active_player):
            # next game:
            next_game = game.forecast_move(move)
            # next game utility:
            utility = self._minimize(active_player, next_game, depth - 1, alpha, beta)
            # pruning test:
            if utility >= beta:
                return utility
            # update max utility:
            max_utility = max(max_utility, utility)
            # update alpha:
            alpha = max(alpha, max_utility)

        return max_utility

    def _minimize(self, active_player, game, depth, alpha, beta):
        """ Implement minimization node in adversarial search with alpha-beta pruning
        Parameters
        ----------
        active_player : object
            Active player that launches minimax search

        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        score : float
            input game utility
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # alpha-beta window test:
        if alpha > beta:
            return float("inf")

        # use score function when maximum depth is reached:
        if depth == 0:
            return self.score(game, active_player)

        # termination test:
        termination_utility = game.utility(active_player)
        if termination_utility != 0:
            return termination_utility

        # do minimization:
        min_utility = float("inf")

        for move in game.get_legal_moves(game.active_player):
            # next game:
            next_game = game.forecast_move(move)
            # next game utility:
            utility = self._maximize(active_player, next_game, depth - 1, alpha, beta)
            # pruning test:
            if utility <= alpha:
                return utility
            # update min utility:
            min_utility = min(min_utility, utility)
            # update beta:
            beta = min(beta, min_utility)

        return min_utility

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

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

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        # time out test:
        if self.time_left() < self.TIMER_THRESHOLD:
            raise SearchTimeout()

        # identify active player:
        active_player = game.active_player

        # identify max utility move:
        max_utility = float("-inf")
        max_move = (-1, -1)

        # able to perform search:
        if depth > 0 and alpha <= beta:
            for move in game.get_legal_moves(active_player):
                # next game:
                next_game = game.forecast_move(move)
                # next game utility:
                utility = self._minimize(active_player, next_game, depth - 1, alpha, beta)
                # update max utility move:
                if max_utility < utility:
                    max_utility = utility
                    max_move = move
                # update alpha:
                alpha = max(alpha, max_utility)

        return max_move
