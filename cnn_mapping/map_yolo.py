"""
Implement the abstract Node class for use in MCTS and the YoloV3TinyRuntime

The yolov3 tiny CNN map is represented as a tuple of 24 values, each either
0, or 1, respectively meaning run on PE1 or run on PE2.
"""

from collections import namedtuple
from random import choice
from monte_carlo_tree_search import MCTS, Node
from simulate_yolov3 import YoloV3TinyRuntime

_YOLOMAP = namedtuple("YoloV3TinyMap", "tup curr_layer terminal, max_runtime")

# Inheriting from a namedtuple is convenient because it makes the class
# immutable and predefines __init__, __repr__, __hash__, __eq__, and others
class YoloV3TinyMap(_YOLOMAP, Node):

    def find_children(ymap):
        # ------------------------------  Add code here ------------------------------#
        # Add your code and modify the return statement below. It should return all the
        # possible children of the current mapping
        return {YoloV3TinyMap(ymap.tup, ymap._get_next_layer(), ymap.is_terminal, ymap.max_runtime)}

    def find_random_child(ymap):
        # ------------------------------  Add code here ------------------------------#
        # Add your code and modify the return statement below. It should return one
        # random child of the current mapping
        rand_0_or_1 = choice(range(2))
        return YoloV3TinyMap(ymap.tup, ymap._get_next_layer(), ymap.is_terminal, ymap.max_runtime)

    def reward(ymap):
        # ------------------------------  Add code here ------------------------------#
        # Add your code and modify the return statement below. Define your reward
        # function in terms of the mapping runtime. It should be a value between 0 and
        # 1, where 1 is the best reward
        return 0

    def is_terminal(ymap):
        return ymap.curr_layer == 22
    
    def _get_next_layer(ymap):
        next_layer = ymap.curr_layer + 1
        if next_layer == 16:
            next_layer = 18
        elif next_layer == 19:
            next_layer = 21
        return next_layer



def main():
    tree = MCTS()
    yolov3 = YoloV3TinyRuntime()
    initialState = tuple([0] * len(yolov3.layers))
    ymap = YoloV3TinyMap(initialState, 0, False, yolov3.max_time)
    
    while not ymap.terminal:
        for _ in range(2):
            tree.do_rollout(ymap)
        ymap = tree.choose(ymap)    

    print(ymap.tup)
    print('End state runtime:', yolov3.get_run_time(ymap.tup))

if __name__ == "__main__":
    main()