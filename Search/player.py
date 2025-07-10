#!/usr/bin/env python3
import time
import math

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR

GLOBAL_TIME_OUT = 0.06
SCORE_WEIGHT = 1
RATIO_WEIGHT = 1


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

    def compute_hash_key(self, node):
        """
        Returns hash key of a given node by using the player's scores, the fish and hooks' positions and fish scores.
        """

        aux = {}

        green_score, red_score = node.state.get_player_scores()

        # fish_pos = (fish_index, (x, y))
        # fish_score = (fish_index, fish_score)
        for fish_pos, fish_score in zip(node.state.get_fish_positions().items(), node.state.get_fish_scores().items()):
            key = str(fish_pos[1][0]) + str(fish_pos[1][1])  # key = "x" + "y" of fish
            aux.update({key: fish_score[1]})

        return str(green_score) + str(red_score) + str(node.state.get_hook_positions()) + str(aux)

    def store_node(self, visited_nodes, node, depth, v):
        key = self.compute_hash_key(node)
        visited_nodes[key] = {'depth': depth, 'heuristic_value': v}

    def compute_distance(self, fish, fish_pos, boat_pos):

        # Obtain information from fishes and boats positions
        fish_x, fish_y = fish_pos[fish][0], fish_pos[fish][1]
        green_x, green_y = boat_pos[0][0], boat_pos[0][1]
        red_x, red_y = boat_pos[1][0], boat_pos[1][1]

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

        # Compute final distance (hook only moves vertically or horizontally)
        final_green_dist = green_dist_x + green_dist_y
        final_red_dist = red_dist_x + red_dist_y

        return final_green_dist, final_red_dist

    def heuristic_function(self, node):  # O(len(fish_position))
        """
        Calculates heuristics of node based on closest fish and the fish value
        @param node
        @return: heuristic_value of current node
        """

        state = node.state

        # Heuristics of green and red
        green_ratio = red_ratio = 0

        # Obtain scores of players
        green_score, red_score = state.get_player_scores()
        # Obtain fish positions
        fish_pos = state.get_fish_positions()
        # Obtain fish scores
        fish_score = state.get_fish_scores()
        # Obtain boats positions
        boat_pos = state.get_hook_positions()

        # Obtain best ratio value for green and red of fish score and fish distance.
        for fish in fish_pos:
            dist_green, dist_red = self.compute_distance(fish, fish_pos, boat_pos)

            if dist_green == 0 or dist_red == 0:
                if fish_score[fish] > 0:
                    return float('inf')
                else:
                    return float('-inf')

            # exp allows to return a value in (0, 1] (1 when dist = 0)
            green_ratio = max(green_ratio, fish_score[fish] * math.exp(-dist_green))
            red_ratio = max(red_ratio, fish_score[fish] * math.exp(-dist_red))

        heuristic_value = SCORE_WEIGHT * (green_score - red_score) + RATIO_WEIGHT * (green_ratio - red_ratio)
        return heuristic_value

    def alphabeta(self, node, depth, alpha, beta, player, starting_time, visited_nodes):

        # Have we surpassed the searching time?
        if time.time() - starting_time >= GLOBAL_TIME_OUT:
            raise TimeoutError

        # Obtain the node's key
        key = self.compute_hash_key(node)

        # If the node is in the transposition table and it contains useful info
        if key in visited_nodes and depth <= visited_nodes[key]['depth']:
            return visited_nodes[key]['heuristic_value']

        # Get children of current node.
        children = node.compute_and_get_children()

        # If we are in a terminal state
        if depth == 0 or len(children) == 0:
            return self.heuristic_function(node)

        elif player == 0:  # Player MAX
            v = float('-inf')
            children.sort(key=self.heuristic_function, reverse=True)
            for child in children:
                v = max(v, self.alphabeta(child, depth - 1, alpha, beta, 1, starting_time, visited_nodes))
                alpha = max(v, alpha)
                if beta <= alpha:
                    break

        else:  # Player MIN
            v = float('inf')
            children.sort(key=self.heuristic_function, reverse=False)
            for child in children:
                v = min(v, self.alphabeta(child, depth - 1, alpha, beta, 0, starting_time, visited_nodes))
                beta = min(v, beta)
                if beta <= alpha:
                    break

        self.store_node(visited_nodes, node, depth, v)

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

        # Starting time of search_best_next_move
        starting_time = time.time()

        # Get children
        children = initial_tree_node.compute_and_get_children()

        # Transposition table
        visited_nodes = {}

        timeout = False
        best_move = None
        current_depth = 0

        while not timeout:

            try:
                # Store heuristic of each child to later know what move was best.
                heuristic_record = []

                for child in children:
                    v = self.alphabeta(child, current_depth, float('-inf'), float('inf'), 1, starting_time, visited_nodes)
                    heuristic_record.append(v)

                # Obtain what the best move is from children
                max_heuristic_value = max(heuristic_record)
                index = heuristic_record.index(max_heuristic_value)
                best_move = children[index].move

                current_depth += 1

            except TimeoutError:
                timeout = True

        return ACTION_TO_STR[best_move]
