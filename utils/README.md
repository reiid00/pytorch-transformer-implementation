# Utils Directory: Essential Utilities for Transformer Implementation

This directory contains a collection of Python scripts crucial for the development of the Transformer model implementation in this project. Each script serves a unique purpose, ranging from defining constants to logging training metrics.

## Table of Contents
1. [Files Description](#files-description)
2. [Contributing](#contributing)

## Files Description

- `constants.py`: This script houses constants required for the Transformer model. It ensures consistency throughout the project by providing a single source of truth for constant values.

- `distributions.py`: This script implements Label Smoothing with KL Divergence Loss, a technique useful for preventing overconfidence and improving generalization in the Transformer model.

- `function_utils.py`: This script is a collection of various utility functions. These include mask generation, checkpoint loading and saving, retrieving training state, counting parameters, calculating BLEU score, evaluating the model and metrics, and batch greedy decoding.

- `logging_tensorboard.py`: This script enables logging to TensorBoard. It contains functions to log various training aspects, including weights, learning rate, loss, and gradients, providing a comprehensive view of the training process.

- `optimizer_n_scheduler.py`: This script deals with optimization and scheduling. It includes the implementation of the Noam Scheduler, an optimizer, and Adam loss with weight decay and AMSGrad.

- `visualize.py`: This script is dedicated to visualization. It provides plotting functions for weights (either individually or in a grid), gradients, learning rate decay, and positional encodings.

## Contributing

Contributions are to this project! To contribute:

1. Fork this repository.
2. Create your feature branch: `git checkout -b feature/your-new-feature`
3. Commit your changes: `git commit -am 'Add some feature'`
4. Push to the branch: `git push origin feature/your-new-feature`
5. Submit a pull request to the `main` branch.

Please ensure your code adheres to the existing style for consistency. 
