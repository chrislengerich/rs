Given a fixed dataset and environmental tools that can be used to collect or generate more data, we would like learn a learning policy:

Notice that we can either: 
1. fix the loss policy and learn a data transformation policy
2. fix the data transformation policy and learn a loss policy.

In practice, humans learn both data transformation policies and loss policies through executive function, but at the 
same time are guided by instrinsic motivation and instincts.

##Research Agenda

### Assume that we'd like to primarily solve the data engineering problem.

* Learn to use stack computation with an API + cognitive dissonance signal [learning_policy_query.md](learning_policy_query.md)
* Show that learning to use stack computation + learning to predict the long-term trajectory encodes improved 
  long-term reward signal.
* Learn to use stack computation with a well-trained world model + cognitive dissonance signal.
* Learn to improve these policies using hindsight RL on trajectories.
* Learn to use stack computation to persist and retrieve data from storage, and bring it onto the stack.
* Learn to use multiple frames of stack to do hierarchical model-based computations.

### Assume that we'd like to learn a loss policy

* Learn a cognitive dissonance signal which is the network's own guess as to it's correctness and the incorrect tokens. 
* Learn something simple, like how how many tokens ahead to attempt to predict using a Transformer