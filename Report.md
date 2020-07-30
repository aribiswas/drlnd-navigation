# drlnd-navigation

## The Environment

The environment used in this project is the Banana environment project in Unity Machine Learning Agents (ML-Agents).

![Alt Text](banana_anim.gif)

A reward of +1 is provided for collecting a yellow banana, and a reward of -1 is provided for collecting a blue banana. Thus, the goal of the agent is to collect as many yellow bananas as possible while avoiding blue bananas.

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around the agent's forward direction. Given this information, the agent has to learn how to best select actions. Four discrete actions are available, corresponding to:

* 0 - move forward.
* 1 - move backward.
* 2 - turn left.
* 3 - turn right.

The task is episodic, and in order to solve the environment, the agent must get an average score of +13 over 100 consecutive episodes.

[1] Environment description from Deep Reinforcement Learning Nanodegree, Udacity.

## The Agent

A Deep Q-Network (DQN) agent combines Q-Learning with a deep neural network to learn a policy. The agent learns by interacting with the environment, receiving rewards and updating the neural network weights. 

The agent in this project uses the **Double-DQN** algorithm which is an improvement on the original DQN algorithm. The double DQN algorithm decouples the action selection from the target network evaluation during computation of the TD error. This reduces issues like overestimation during training. A full description of the Double-DQN algorithm can be found in the paper [here](https://arxiv.org/abs/1509.06461).

The goal of the agent is to maximize the discounted return $$G=\sum_{k=t}^{T}\gamma^{k}r_{t+k}$$

<pre><code>
</code></pre>

### Deep Q-Network model

A simple neural network with three fully connected layers is used in this project. The fully connected layer outputs are passed through ReLU layers during computing forward pass. To stabilize training, a batch normalization layer wraps the first fully connected layer.

<pre><code>class QNetwork(nn.Module):
    
    def __init__(self,osize,asize,seed=0):
        
        super(QNetwork,self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        # define the deep neural network structure
        self.fc1 = nn.Linear(osize,64)
        self.fc2 = nn.Linear(64,32)
        self.fc3 = nn.Linear(32,asize)
        self.bn1 = nn.BatchNorm1d(64)
        
    
    def forward(self,x):
        
        # make forward pass through the network
        x = self.bn1(F.relu(self.fc1(x)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
</code></pre>

### Agent Hyperparameters

* The DQN algorithm relies on the ***epsilon*** hyperparameter for exploration. For this project, the ***epsilon*** parameter is initially set to 0.95 at the begininng of training when the policy favors random actions. At each agent step, the epsilon value is decayed exponentially by a decay factor of 1e-5, with a minimum epsilon limit of 0.1. With this exponential decay, the agent favors exploration towards the beginning of training and exploitation later.
* A discount factor of 0.99 favors long term rewards during return computation.
* The agent learns from mini batches of 64 experiences with the adam optimizer at a learn rate of 2e-4.
* The target Q-network is soft-updated with an update factor of 1e-3

<pre><code>BUFFERSIZE = int(1e6)    # Experience buffer size
GAMMA = 0.99             # Discount factor
EPSILON = 0.95           # Epsilon parameter for exploration
DECAY = 1e-5             # Epsilon decay rate
EPMIN = 0.1              # Minimum value of epsilon
MINIBATCHSIZE = 64       # Batch size for sampling from experience replay
LEARNRATE = 2e-4         # Learn rate of Q network
TAU = 1e-3               # Target network update factor
</code></pre>



