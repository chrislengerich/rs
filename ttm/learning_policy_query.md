We parameterize a learning policy as:
1. a set of batch error examples (spans of state) which were are used to motivate the policy (ie. improve <X>, predict 
   <X>, control <X>)
2. a set of labeled examples, where each example has the parameterization x: (goal, state, question) and y: (response)
3. a set of trajectories which ground the policy to its utility to achieve the goal
4. a training policy, $\theta_l$ which is used to load the policy into the agent. This is either training via 
   predictive loss or by pushing the trajectories onto the agent's stack. Borrowing terminology, we can call this 
   "Learning, Fast and Slow"

*Steps:*
1. Write down a human-guided trace of a learning policy (examples).
2. A symbol to label the learning policy and parameterize it.
3. Write down some human-guided trajectories which use the symbol of the learning policy and ground it to utility.
4. Pre-train the human-guided trajectories into the dataset.

*Data representation:*
1. pickle file.
2. flat files: 
   1. error examples.
   2. prompts
   3. name
   4. grounded trajectories using the name + goal
   5. training policies for adding this behavior to the agent

Experiments: 
    1. Reward (intrinsic and otherwise)
    2. Reasoning with stories (motivated and unmotivated)

4. Conceptually, we are taking selected information from our current state and selected information from our past 
   history or our internal world model and synthesizing this into a compactly expressed policy which drives downstream 
   utility, using a stack as scratch space. This is similar to key-value lookup operations, however: 
   1. We work with natural language text, for which we have abundant pre-training data and for which behavior 
      policies are easier to analyze.
   2. We are motivated to do this by the meta-learning of intrinsic motivation + rewards (play, curiosity, learned 
      rewards).
   3. We describe using APIs and sampling from our own internal world models together.
   4. We describe a fast meta-learning operation which lets us construct a sampling policy from our world simulator 
      from only a handful of examples.

We note that, since a learning policy is simply text (base case), the response can itself be a learning policy (the 
inductive case). Therefore, we can learn to generate arbitrarily deep nested learning policies which can 
parameterize a tree structure of learning policies.

We generate a new policy by sampling from the policy distribution based on the current state and 
experiment to ground the policy to its utility. In the pre-training phase, we provide human scaffolds
via labeled training examples and templated code and provide several complete examples of traces of 
learning policies as meta-pretraining examples.

Repeat this 2 more times, so we have 3 examples of the policy of synthesizing a learning policy.
These become the examples for the higher level-policy (1) which let us invoke this learning subroutine again. 
We also would like to write a trace of doing this within our higher context, which (with a lookup/indirection table),
will ground the invocation of this policy to utility. When the learner invokes the learning policy (parameterized by 
the goal), the learning subroutine is invoked.

Key challenges:

1. Abstraction/compositionality -> provided by the lookup table. We have learned and labeled compression functions, 
   which are trained by prediction. So the goal is: 
   a. to efficiently learn a compression function to create a new label for spans of your input data.
   b. train a cognitive dissonance function to learn to use those existing compression functions + create new 
   compression functions on the fly.
   c. one example of a compression function is to compress several spans of text into a smaller span. This also, 
   however, works as a bijective mapping - the smaller span can stand in for the other spans and is weakly 
   associated with them.
   d. the most important compression functions are those which compress your current state into a behavior which is 
   useful for survival, as in prompt engineering.
   e. we provide one such example of useful compression functions of your environment (knowing which questions asked 
   to ask to a language model in order to load the correct information into your stack).
   f. we provide an example of learning to compress - learning to synthesize + name new behaviors from a small 
   number of examples, experiment to ground them to utility, and then learn from those experiments.

Consciousness: 
    1. Compression guided by cognitive dissonance and labeled via grounding to instrinsic rewards.
    2. Learning by prediction + outside reinforcement.

2. To do this effectively, Codex needs to learn compositional abstractive operations interpreted as 
   executive function, just based on the process of next-token prediction. This allows for soft retrieval and a true 
   compositional structure of reasoning and thought.
3. If the structure of the trajectories/training data was stack-structured, then we could use tree-based lookups to
   descend into prior policies (and importantly, to recurse out of them). Would want to also meta-learn a 
   compressive/abstraction policy for iterating over each of these learning policies.

Load all learning traces into the agent, with this taken to be a higher-level action.

At the higher level, everything can learned by invoking predictive learning routines on a dataset of text.
Specifically what text we write (traces of cognition, internal stories, etc.), and how we ground them to other 
parts of the text (behaviors, etc.) is the key question. It can be permuting the inputs together or it
can also be other things.

Outer loop: 
   Ground to reward.
Inner loop: 
   Learn to predict.
Actions: 
   Zoom in (trace though a portion of the calculation on your stack)
   Zoom out (take several buffers of examples and learn from them)
Everything can be represented as text. Reward grounding and representations are meta-learned, and in particular, 
prediction error and satiation curves is a meta-learning signal guiding attention.

State -> action
State, action -> State_{s+1}, predicted_reward
t = t + 1

State_{s+1} includes capabilities of text-based retrieval, and text-based compression.

we train via tracing policies using traces of human policies (thoughts) expressed in text and densely labeled with 
intrinsic rewards. We decode via maximium-likelihood decoding subject to the goal of predicted_reward = infinity.
We label the traces with intrinsic rewards to go.

Key challenges: 
1. Keeping a meaningful representation of state in memory to keep it tractable (compression and retrieval).
2. Long-term grounding of rewards.
3. Learning of intrinsic motivation functions.

General flow:
   Goal -> Question -> Examples (labeled internally or externally) -> Answer

Next steps: 
1. Create a policy which can learn to predict that it's not doing well given a trace of its execution.
2. Create a policy which can learn to: 
   1. Invent a new behavior. []
   2. Tweak an existing behavior.
   
Would like to meta-learn a function for attention over prediction error which could be considered to be that of 
cognitive dissonance.

