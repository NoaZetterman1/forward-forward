### Neural Network Layers

This repository contains implementations of the [Forward-Forward Algorithm](https://arxiv.org/abs/2212.13345), as well as extensions to use recurrency within the paradigm.

Furthermore, a small ChatGPT generated toy dataset is included under datasets.


### Setup
Using Python 3.10, run `pip install -r requirements.txt`. To train a model, modify main.py to use the structure and dataset of choise and run main.py. To use Wandb, a project must be created at their website. However, with some minor modifications it is also possible to run the code without Wandb (comment out the logging in the networks as well as wandb initialization). 

In the folder greedy_pretraining, there is an implementation to train with Forward-Forward pre-training and backpropagation fine-tuning, which has its own main file that can be ran for training.