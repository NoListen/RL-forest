# Reinforce
Reinforce is one of the most simple Policy Gradient.

The training signal is derived directly from the Monte-Carlo Return.

# Enviroment

The environment used here is one two-agents pong game based on pygame engine.

To make the learning process easier, the scores are removed.

You can add the scores by uncommenting the `drawScore` function

# Implementation

It adjusts the same architecture like the numpy implementation in Karpathy's [blog](https://gist.github.com/karpathy/d4dee566867f8291f086), which only applies mlp network with only one hidden layer.

# TODO

Mlp is hard to generalizes and conv network with max pooling layers may learn faster. 
