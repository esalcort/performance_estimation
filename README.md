# Performance Estimation

Performance estimation exercises using machine learning

## Contents
* **gemm_runtime_estimation/**
    - **gemm_nn.c**: A dummy matrix multiply code with variable input sizes M, N, and K
    - **sample.csv**: A set of profiling results after running gemm_nn with different inputs
    - **estimation_models.ipynb**: A jupyter notebook with examples to train ML models that predict data in sample.csv
* **cnn_mapping/**
    - **monte_carlo_tree_search.py**: A minimal implementation of MCTS imported from [1]
    - **tictactoe.py**: An example application of MCTS imported from [1]
    - **simulate_yolov3**: A class that estimates computation and communication runtime of a yolov3 tiny CNN running on two processors
    - **map_yolo.py**: A class that implements the MCTS Node class for the simulate_yolov3 file

## References
[1] Luke Harold Miles, monte_carlo_tree_search.py, Github Gist, https://gist.github.com/qpwo/c538c6f73727e254fdc7fab81024f6e1

## Contact
If you have questions or suggestions about this repository, or a problem running examples, you can contact Susy at esalcort@utexas.edu
