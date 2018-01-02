# Isolation-Playing Agent through Adversarial Search

![Example game of isolation](viz.gif)

## Adversarial Search Implementation

### Min-Max Search

The min-max search agent is implemented as follows.
The implementation follows the <a href="https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md">AIMA Pseudocode</a>

---
__function__ MINIMAX-DECISION(_state_) __returns__ _an action_  
&emsp;__return__ arg max<sub> _a_ &Element; ACTIONS(_s_)</sub> MIN\-VALUE(RESULT(_state_, _a_))  

---
__function__ MAX\-VALUE(_state_) __returns__ _a utility value_  
&emsp;__if__ TERMINAL\-TEST(_state_) __then return__ UTILITY(_state_)  
&emsp;_v_ &larr; &minus;&infin;  
&emsp;__for each__ _a_ __in__ ACTIONS(_state_) __do__  
&emsp;&emsp;&emsp;_v_ &larr; MAX(_v_, MIN\-VALUE(RESULT(_state_, _a_)))  
&emsp;__return__ _v_  

---
__function__ MIN\-VALUE(_state_) __returns__ _a utility value_  
&emsp;__if__ TERMINAL\-TEST(_state_) __then return__ UTILITY(_state_)  
&emsp;_v_ &larr; &infin;  
&emsp;__for each__ _a_ __in__ ACTIONS(_state_) __do__  
&emsp;&emsp;&emsp;_v_ &larr; MIN(_v_, MAX\-VALUE(RESULT(_state_, _a_)))  
&emsp;__return__ _v_  

```python
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
```

### Alpha-Beta Search and Iterative Deepening

The min-max search agent with alpha-beta pruning and iterative deepening is implemented as follows.
The implementation follows the <a href="">AIMA Pseudocode</a>

---
__function__ ALPHA-BETA-SEARCH(_state_) __returns__ an action  
&emsp;_v_ &larr; MAX\-VALUE(_state_, &minus;&infin;, &plus;&infin;)  
&emsp;__return__ the _action_ in ACTIONS(_state_) with value _v_  

---
__function__ MAX\-VALUE(_state_, _&alpha;_, _&beta;_) __returns__ _a utility value_  
&emsp;__if__ TERMINAL\-TEST(_state_) __then return__ UTILITY(_state_)  
&emsp;_v_ &larr; &minus;&infin;  
&emsp;__for each__ _a_ __in__ ACTIONS(_state_) __do__  
&emsp;&emsp;&emsp;_v_ &larr; MAX(_v_, MIN\-VALUE(RESULT(_state_, _a_), _&alpha;_, _&beta;_))  
&emsp;&emsp;&emsp;__if__ _v_ &ge; _&beta;_ __then return__ _v_  
&emsp;&emsp;&emsp;_&alpha;_ &larr; MAX(_&alpha;_, _v_)  
&emsp;__return__ _v_  

---
__function__ MIN\-VALUE(_state_, _&alpha;_, _&beta;_) __returns__ _a utility value_  
&emsp;__if__ TERMINAL\-TEST(_state_) __then return__ UTILITY(_state_)  
&emsp;_v_ &larr; &plus;&infin;  
&emsp;__for each__ _a_ __in__ ACTIONS(_state_) __do__  
&emsp;&emsp;&emsp;_v_ &larr; MIN(_v_, MAX\-VALUE(RESULT(_state_, _a_), _&alpha;_, _&beta;_))  
&emsp;&emsp;&emsp;__if__ _v_ &le; _&alpha;_ __then return__ _v_  
&emsp;&emsp;&emsp;_&beta;_ &larr; MIN(_&beta;_, _v_)  
&emsp;__return__ _v_  

```python
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
```

## Heuristic Analysis

### Heuristics Implementation

Here three extra heuristics are implemented.

The **first** one is just a simple modification of  the improved heuristic.
Here L2 norm is used to encourage the agent to pursuit a larger margin of the number of legal moves between active and opponent players.

```python
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
```

The **second** one is also a simple modification of the improved heuristic.
Here L3 norm is used to encourage the agent to pursuit a larger margin of the number of legal moves between active and opponent players.

```python
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
```

The **third** one combines the ideas from improved and center heuristics.
Here a combined score is calculated from both the difference of number of legal moves and center score
between active and opponent players.

```python
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
```

### Performance Analysis

The performance analysis carried out three times and the results are as follows:

**Tournament 1**
```shell
                        *************************                         
                             Playing Matches                              
                        *************************                         

 Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3
                        Won | Lost   Won | Lost   Won | Lost   Won | Lost
    1       Random      10  |   0    10  |   0     9  |   1     8  |   2  
    2       MM_Open      6  |   4     6  |   4     7  |   3     6  |   4  
    3      MM_Center     6  |   4     9  |   1     7  |   3     5  |   5  
    4     MM_Improved    6  |   4     6  |   4     6  |   4     5  |   5  
    5       AB_Open      5  |   5     6  |   4     3  |   7     6  |   4  
    6      AB_Center     7  |   3     6  |   4     5  |   5     6  |   4  
    7     AB_Improved    4  |   6     4  |   6     5  |   5     5  |   5  
--------------------------------------------------------------------------
           Win Rate:      62.9%        67.1%        60.0%        58.6%    
```

**Tournament 2**
```shell
                        *************************                         
                             Playing Matches                              
                        *************************                         

 Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3
                        Won | Lost   Won | Lost   Won | Lost   Won | Lost
    1       Random       7  |   3     7  |   3     7  |   3    10  |   0  
    2       MM_Open      6  |   4     7  |   3     7  |   3     7  |   3  
    3      MM_Center     6  |   4     5  |   5    10  |   0     8  |   2  
    4     MM_Improved    6  |   4     6  |   4     6  |   4     5  |   5  
    5       AB_Open      5  |   5     5  |   5     5  |   5     7  |   3  
    6      AB_Center     7  |   3     7  |   3     6  |   4     6  |   4  
    7     AB_Improved    5  |   5     6  |   4     6  |   4     4  |   6  
--------------------------------------------------------------------------
           Win Rate:      60.0%        61.4%        67.1%        67.1%    
```

**Tournament 3**
```shell
                        *************************                         
                             Playing Matches                              
                        *************************                         

 Match #   Opponent    AB_Improved   AB_Custom   AB_Custom_2  AB_Custom_3
                        Won | Lost   Won | Lost   Won | Lost   Won | Lost
    1       Random       8  |   2     9  |   1     8  |   2    10  |   0  
    2       MM_Open      5  |   5     6  |   4     9  |   1     8  |   2  
    3      MM_Center     7  |   3     6  |   4     7  |   3     8  |   2  
    4     MM_Improved    6  |   4     8  |   2     5  |   5     8  |   2  
    5       AB_Open      4  |   6     7  |   3     7  |   3     4  |   6  
    6      AB_Center     6  |   4     5  |   5     5  |   5     4  |   6  
    7     AB_Improved    5  |   5     7  |   3     5  |   5     4  |   6  
--------------------------------------------------------------------------
           Win Rate:      58.6%        68.6%        65.7%        65.7%    
```

The first custom heuristic, the one using L2 norm, outperforms the baseline one 3 out of 3 times.

I think the first custom heuristic should be used for the following reasons:

1. Steadily improved performance over the baseline heuristic.
2. No extra computing overhead(compared with heuristics based on complex human knowledge).
3. No hyper-parameter tuning(compared with the third custom heuristic).
