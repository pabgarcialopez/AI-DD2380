#!/usr/bin/env python3
import random
import time

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
import math

class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def compute_distance(self, fish, fish_position, boat_position):

        # Obtain information from fishes and boats positions
        fish_x, fish_y = fish_position[fish][0], fish_position[fish][1]
        green_x, green_y = boat_position[0][0], boat_position[0][1]
        red_x, red_y = boat_position[1][0], boat_position[1][1]

        # Direct distances from green and red to fish
        green_dist_x, red_dist_x = abs(fish_x - green_x), abs(fish_x - red_x)
        green_dist_y, red_dist_y = abs(fish_y - green_y), abs(fish_x - red_y)

        # Take into account whether one boat blocks the other:

        # If red blocks green
        if fish_x < red_x < green_x or green_x < red_x < fish_x:
            green_dist_x = 20 - green_dist_x  # Boat has to go the other way around

        # or if green blocks red
        elif fish_x < green_x < red_x or red_x < green_x < fish_x:
            red_dist_x = 20 - red_dist_x  # Boat has to go the other way around

        # Compute final distance (Pitagoras)
        final_green_dist = math.sqrt(green_dist_x * green_dist_x + green_dist_y * green_dist_y)
        final_red_dist = math.sqrt(red_dist_x * red_dist_x + red_dist_y * red_dist_y)

        return final_green_dist, final_red_dist

    def heuristic_function(self, state):  # O(len(fish_position))
        """
        Calculates heuristics of node based on closest fish and most valuable fish
        @param state - state of current node
        @return: heuristic_value of current node
        """
        green_score, red_score = state.get_player_scores()
        heuristic_value = (green_score - red_score)

        return heuristic_value

    def alphabeta(self, node, depth, alpha, beta, player, max_allowed_depth):

        
        # If we are in a terminal state
        if depth == max_allowed_depth or len(node.state.get_fish_positions()) == 0:
            v = self.heuristic_function(node.state)

        elif player == 0:  # Player MAX
            v = float('-inf')
            children = node.compute_and_get_children()
            for child in children:
                v = max(v, self.alphabeta(child, depth + 1, alpha, beta, 1, max_allowed_depth))
                alpha = max(v, alpha)
                if beta <= alpha:
                    break

        else:  # Player MIN
            v = float('inf')
            children = node.compute_and_get_children()
            for child in children:
                v = min(v, self.alphabeta(child, depth + 1, alpha, beta, 0, max_allowed_depth))
                beta = min(v, beta)
                if beta <= alpha:
                    break

        return v

    def search_best_next_move(self, initial_tree_node: Node):
        """
        Use minimax (and extensions) to find the best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        # Starting time of searching
        start = time.time()

        children = initial_tree_node.compute_and_get_children()

        best_move = None
        best_value = float('-inf')

        # Iterative deepening in terms of time. Timeout occurs at 0.8-0.9s spent approx.

        # Time allowed to search. It gets incremented by sum_time in each iteration.
        time_threshold = 0.001  # This is initialized to 0.001 allow search of depth 1
        # sum_time = [0.005, 0.025, 0.050, 0.070, 0.1, 0.2]
        time_out = 0.3
        i = 0
        max_allowed_depth = 3

        while time.time() - start < time_out:
            for child in children:
                start_alphabeta = time.time()
                v = self.alphabeta(child, child.move, float('-inf'), float('inf'), child.state.get_player(), max_allowed_depth, start_alphabeta, time_threshold)
                if v > best_value:
                    best_value = v
                    best_move = child.move

            i += 1
            print(f"Time spent up until iteration i = {i}: {time.time() - start}. Threshold: {time_threshold}")
            time_threshold += 0.01

        return ACTION_TO_STR[best_move]