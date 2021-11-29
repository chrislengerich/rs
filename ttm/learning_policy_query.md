*AIs:*
a. Clean up the current experiment.

a. Train an online value policy over rollouts using predictive error (consciousness analog).
   Use human-labeled flat learning policies to permute the stack state (in process). Train for reward prediction and
   downstream prediction error.
   Goal: show that we can get the model to accurately predict future reward, decode to future reward and learn to 
select specific trajectoried onto its buffer.

b. Composite (multi-step) trace prediction:
   Train the above but use fixed human-labeled composite learning policies (in process).
        a. Train with flattened policies. 
        b. Train with policy traces.

b. Composite trace prediction:
   Train the above but use fixed human-labeled composite learning policies (in process).
        a. Train with flattened traces. 
        b. Train with indirect traces.

b. Show that training a fast attention-based experimental policy over rollouts can drive the dynamic exploration of
   of complex behaviors within the manifold of the latent space, running on a neural computer.
d. Pass level 2, write up a revised manuscript based on TextWorld, send it out to review committee by Wednesday.
e. Share with Gradient review committee, share with academic review committee (+ add in several professors who you 
might want to apply to a PhD with + Volo)
f. Push the research note to arXiv by Friday.










































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

Two loops, running simultaneously:
   Ground to reward.
   Learn to predict.
Actions: 
   Give examples of an abstraction.
   Abstract (take several buffers of examples and learn from them)
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

Core principles of human online learning:
      
   4. Behavior policy has two types - a learning policy and a value policy:

      1. Learning policies are expressed as a small number of examples (trajectories of tokens) rather than a large 
         number of examples of bits with labels. A well-trained neural network with 
         grounded predictive loss from tokens is a neural computer which can execute these policies conditioned on a 
         stack state. This learning policy rewrites the trajectories on your stack.
      2. We also have a value policy which uses soft attention over all trajectories currently in the stack, including 
         itself. It outputs a value prediction for that stack state, which is compared to the value achieved after a 
         learning policy is invoked to produce a reward prediction error derived from the environment.
   
   5. The set of trajectories on your stack (roughly those you've perceived + those you've imagined) + your learned 
      value policy completely determines your behavior, modulo quantum interactions.
   
   6. Trajectories are stored on the stack, and also in long-term memory. For the purposes of this conversation, we 
      only consider a very large stack, but note that (key,value)-based retrieval and forgetting policies over 
      trajectories are important for scaling out a practical agent. We note that a Transformer would likely be similar to the 
      attention mechanism in the Transformer used in stack transformations.
   
   7. External value hard-coding (via evolution):
          1. Have a well-trained value function (using dense training methods like hindsight labeling).
          2. Hard-code your value prediction to be high when sampling an action.
          3. Remove individuals from the population which are not high-value/have opted for a value function which 
             is not well-calibrated to reality.
   
   8. We frame action as communication and derive insights from the duality, since action takes place 
      completely within the model's perceptual world (ie. Plato's cave).

   9. We learn a value policy (intrinsic motivation) which describes attention over trajectories.
   
      1. Behaviors are used because they are valuable. They are valuable because they reduce reward prediction error 
         and prediction error/uncertainty. At inference (which is also running constantly), we are hardwired by 
         evolution to decode behaviors generally to high-reward trajectories.
      2. We define these simply as a learned value function over the stack of trajectories. Starting from the current trajectory state, we can take ANY 
         learning policy which is expressed in code + prompts and generate it. If it improves the ROI w.r.t to 
         environmental fitness (high reward, minimal space) then we keep it, otherwise, we throw it away (or in a 
         softer form), we retrieve it less or more based on the current state. We allow for policies which change 
         the ROI estimation of specific parts of the stack (learning policies, trajectories, etc.). This ROI 
         estimation/soft retrieval helps perform sharp credit attribution of the stack segments to the longer-term 
         reward and itself is trained based on prediction of ROI given training examples of stacks and their associated ROIs, and is
         part of the running code.
   
        Key problems:
         1. How do you transmit an idea that's valuable, which causes people to pay attention? (specifically, 
            high-contrast with the existing value policy (ie. would result in contrast). Attention over data <-> 
            value. Value w.r.t. to satiation curves, value w.r.t. to generalization, which is heavily recency-weighted.
            1. Satiation curves.
            2. Reduces cognitive dissonance.
            3. Reduces cognitive dissonance w.r.t. my goals.
            4. Value function for planning is often related to comparing two trajectories: 
               1. Pair 1:
                  a. What I expected to happen. What did happen. Goal: Minimize loss here.
               2. Pair 2:
                  b. What did happen. What would have happened if some part of the state was otherwise.
               3. Pair 3:
                  c. My resources. My resources within the next state. Goal: Maximize.
            5. There's value over specific trajectories (I have a policy which leads to) as well as over pairs of 
               trajectories.
            6. Ways to learn these learning policies.
               1. Imitation learn the learning policies. Ground through use by embedding in trajectories.
               2. Apply gradient descent or any other learning algo using the inverse of your value function as your 
                  loss.
               3. Past state, value -> next state (including the trace of the transformation), value (collect a 
                  bunch of these pairs of data, then do inference by predicting a high value and using a neural network to 
                  to fill in the transformation based on a RL-based exploration/exploitation policies). Learn to 
                  predict some high-value transformations of the trajectories.
               4. *Goal:*
                  1. Predict the value of my stack and compare to the actual value for many different stacks (during 
                     training).
   
   10. Learning policy is a transformation of the stack parameterized by a short name, code and data (which takes the 
       stack as input, compresses it along with concatenating to other parts and outputs a new stack). It is learned 
       through grounding examples and can be executed on a neural computer or a classical computer. Traditionally, 
       learning policies are used to reduce prediction error, which is used to reduce reward prediction error.

        1. Planning: Counterfactual reasoning/imagination is basically the primary driver of planning.
           1. What would happen next? 
           2. What must have happened to cause this?
           3. A prediction agent is an agent with imagination. 
           4. An agent with imagination is a planning agent.
        2. Learning to abstract is a dual problem of learning to generate examples == learning. Question synthesis can 
           be an emergent behavior of counterfactual reasoning, motivated by reward. It can also be considered to be (key,value)
           -based retrieval:
           1. What questions would be helpful here?
              1. Simplification.
              2. Concreteness.
              3. Similarity (retrieval/rewriting policy)
           2.  The process of communication is the bijection of abstraction -> examples -> abstraction.
           3. How do we encode them so that the abstraction of the policy is reliably passed to the agent and can be 
              invoked for the state?
        3. Goal direction and divide-and-conquer.
           1. Learnable prediction loss.
           2. Learnable imagination.
        4. Simplification.
           1. How can I express this more simply?
        5. Hindsight labeling.
        6. Prediction error for the value function:
           3. Prediction error == cognitive dissonance -> drives use of learning policies to rewrite the stack (which 
              may include the learning code that it's currently running) and learning quickly based on this.
        7. Hindsight compression:
           1. Rewriting a stack into a shorter version of what happened or changing representations into simpler, more 
              compact forms so that imagination becomes more tractable.
        8. Weighted stack retrieval based on the value policy.
           1. Either which trajectories to load, then sample an action (or more directly), which learning policy to run.
        9. In the general form, a learning policy is expressed as a generated set of examples which apply a 
           transformation to our stack (code + prompts). We note that by expressing a tokenized representation and 
           by only requiring a handful of examples, we can push this policy onto the stack cheaply, allowing it to be

   11. Training process:
       1. Imitation-learn/copy a lot of learning algos from classical CS expressed a tokens with a handful of examples.
       2. Aim to learn a cognitive dissonance function (either prediction error, reward prediction error or 
          confusion) that will drive use of of stack-conditional learning algos starting from these and new simplified 
          facts and be useful for online learning.
          1. Meta-learning a cognitive dissonance function may be possible without having access to the labels, just 
             based on some type of grounded confusion.
       3. Estimate the value of the stack with learned RPE policy w.r.t. to the traces of execution, guided by killing 
          miscalibrated stacks.
      
   15. TL;DR:
       1. We are a constantly running stack of **value-weighted trajectories** which can be run on classical computers 
          and neural computers. The value-weighting is attention over the traces of trajectories which is learned from 
          hardcoded RPE and PE. In the case where the trajectory is the trace, we can learn these directly.
   
   Human behavior explanation/metaphor:
      
   13. Strange loop explanation:
       1. We are constantly rewriting the code that we're currently running.
       2. In particular, we are rewriting our value policy and the current trajectories on our stack, so that our 
          stack becomes high-value, yet compact.
       3. Figuring out consciousness is probably not the most important problem in the world, but it's a good start 
          to developing strong value functions.
   14. Fame:
          1. Learning that someone is successful from a handful of examples, not just one (ie. learning to generalize 
             from a situation).
   15. Communication:
       1. Some people learn quickly (from a few examples), some learn slowly (from many), some don't learn at all. 
          The more similar your NNs are, the fewer number of examples that you need to transmit.
   16. Goals: 
       1. A goal is a change to your state which forces you to have cognitive dissonance if reality does not match 
          your goal. In general, humans are meta-learned to have a purpose (a goal).
   17. Consciousness:
       1. Is just a learned meta-learning policy that governs the revision of trajectories guided by 
          imagination error (prediction error relative to current perceptions), along with cognitive dissonance.
   18. Planning:
       1. Is just imagination of a future state and filling in between the current and the next states using 
          log-likelihood along with reward prediction (a specific way of invoking your world model).
   19. Generalization:
       1. Any abstract policy can be learned in generality through a sufficient number of examples which 
          are concrete and specific but sufficiently contrasting from each other.
       

General-purpose online learning algo:

   -> Retrieval is the inverse of generalization.

   -> Trajectory/Decision Transformers train (implicitly) via RPE.
   -> A high-value stack is a stack of trajectories which have high reward or have been
      associated with high RPE or prediction error (just hard negative/positive mining).

   20. Learning to abstract in on online setting is generally based on getting new data:
       1. At every step, make a prediction/inference of input and output at many timescales.
       2. At the next step, compare to the incoming sensory input. When loss is high:
          1. Generate some similar examples given a goal.
             1. Retrieve similar examples from storage.
             2. Try again/retry.
             3. Use metaphor/analogy.
             4. Break a problem into parts.
             5. Ask questions that are relevant to pull information onto the stack.
          2. Place that example on the stack, label it based on the actual input.
          3. Generate a simplification of the stack or imagine some new policies beforehand.
       3. Set a new goal and prediction (by imagining decoding to high reward using this data).

   21. Load the learning trajectories in statically based on similarity.
   22. Load them in based on RPE or prediction error.
   23. Synthesizes new policies based on the stack.
       
   24. Learning to value trajectories -> based on RPE and R (indirectly):
       1. Use hindsight labeling.
       2. Learn a labeled trajectory comparison function.
       3. Grounding via returns to go.
       4. Ground via returns.

   
    

   

