# Quixo

## Description
Quixo is a board game for two players where the goal is to be the first player to arrange five of their pieces in a row, either horizontally, vertically, or diagonally.
The board is a 5x5 grid of squares and starts with all 25 squares empty.
On each player turn, they choose one of the emtpy squares or one of their own pieces, turn it to the player's symbol and move it on one of the four sides of the board, pushing all the pieces in that row one square.

## DQN player
The player is implemented as a DQN agent, so it approximates the Q function using a neural network and the Q values are used to select the best action to take in a given state of the game.

### Neural network
The neural network is implemented as a fully connected network with 3/4 hidden layers (one is optional) and ReLU activation functions. The input space is the `N*N` board, where each square has the possible values `EMPTY`, `X` and `O`. The output space is dependent on the `ACTION_SPACE` that can include all the possible actions, also the invalid ones e.g. `(0, 0) TOP` or can include only the valid ones e.g. `(0, 0) BOTTOM`.

### Training
The network is trained using the Adam optimizer and the loss function is the mean squared error between the target Q values and the predicted Q values.
To reduce variance in the training are used two networks, one for the target Q values: `target_net` and one for the predicted Q values: `policy_net`. The `target_net` is updated after each optimization step with the weights of the `policy_net` rescaled by a factor `TAU`.

### Policy
During training, the agent is using an epsilon-greedy policy to select the next action, where the epsilon cloud be a costant value `EPSILON` or a decaying value as `EPSILON_B` / (`EPSILON_B` + n. of steps). The espilon action is chosen as a random action belonging to the `ACTION_SPACE` without the best action, where each action is  weighted by the q values of the `policy_net` and normalised with softmax.
Instead, during testing, the agent uses a greedy policy to select the next action, where the greedy action is the best action belonging to the `ACTION_SPACE`.

### Game batch
The agent samples a batch of `BATCH_SIZE` games with their trajectories (state, action, reward) and then calculates the mean squared error between the expected Q values and the predicted Q values. The expected Q values are calculated using the `target_net` and the Bellman equation.

### Invalid moves
The player implements a trace of the last moves done in a specific state to avoid repeating the same move, that could be invalid, chosing the second, third, etc. best action.

### Board normalization
The board of each state is normalized to a canonical form that exploits simmetries if `TRANSFORMATION` is set to `True`. The board is transformed in all the possible ways and the one with the lowest hash value is chosen as the canonical form and also the moves are tranformed coerently with the board. To speed up the process the most used transformations are stored in a dinamic dictionary used as a cache.

## Environment
The main goal of the environment is to simulate a match between the agent and a environment player that can be chosen e.g. the `RandomPlayer`, The game is suddivided in steps, where each step is a turn of both player. The environment takes the action of the agent and if it's valid return the next state, the reward and if the game is ended. The next state is the game board after the agent and the environment player have done their moves.

## Last move player
An other type of player is the `LastMovePlayer` that is based on any other player's strategy but it analyses the board to find if there is a move that can win the game or if the opponent can win the game in the next move, in this case it blocks the opponent if it's possible. In the other cases it plays as the base player.

### Masks
To find which states of the game are winning or losing in an efficient way it checks if the mask of the board is in the precomputed dictionary of the terminal masks. The mask is board where only the squares of the potential winning lines are set to 1, the other squares are set to 0, in this way many games are equivalent and can be grouped in the same mask.
From the dictionary it gets the winning or losing square (to archive a winning line) that is used to evalute the move to win according with specific board.
To avoid losing the player uses the base player to find the best move and then it checks if the opponent can win in the next move, if it's the case an other move is chosen.

### Role
The `LastMovePlayer` can be used as agent or as environment player, in the first case the base player can be trained as usual, in the second case the agent it's trained and tested against a more challenging player.

## Training
During the training many combinations and hyperparameters have been tested that can be summerized as follows:
- `N` - The board size, to understand how the agent performs with different complexity of the game.
- `ITERATIONS` - The number of iterations, that reach performance plateau around 1000K iterations.
- `VERSION` - The version of the agent, the first version is trained against only the `RandomPlayer`, from the second version it's trained against `RandomPlayer` and all previous version of the `DQNPlayer` with same hyperparameters.
- `TRANSFORMATION` - The board is normalized to a canonical form after each move to reduce the state space, the process is slow and don't improve the performance.
- `INVALID_SPACE` - The action space is expanded with all possible actions, also the invalid ones that can be done in every state. In this way the agent can learn that some actions are invalid and avoid them but increase the state space and the training complexity.
- `INVALID_MOVES` - The environment notify the agent when it does an invalid move instead of terminating the game, so the agent can do another one according to the policy.
- `LAST_MOVE` - Both the agent and the environment player use the `LastMovePlayer` and the base player is chosen according to other parameters.
- `LOAD` - The environment players are loaded from two different setups, the first is `simple` and it's the standard one, so the agent is trained/tested against the `RandomPlayer` and it's previous versions. The second is `mix` and the agent is trained/tested against the `RandomPlayer` and each player already trained among the 100K and 1000K iterations.
  
### Network parameters
- Rewards of each action: the rewards of each action are tested between [0, 1] and [-1, 1], with different values for `MOVE_REWARD` in range [0, 0.05]
- Epsilon mode: `ESPILON_MODE` chose the epsilon value between `EPSILON` and `EPSILON_B` / (`EPSILON_B` + n. of steps).
- Layers: the number of layers of the network can be 3 or 4 and the number of neurons per layer are tested in the range [128, 1024].
- Batch size: `BATCH_SIZE` is tested in the range [2, 64].
- Gamma: `GAMMA` is the discount factor and it's tested in the range [0.5, 0.9]
- Tau: `TAU` is the factor used to update the `target_net` and it's tested in the range [0.01, 0.05]
  
## Results
The agent is trained to win against as many players as possible, so it tries to find the best and more general policy to win the game.

### Usage
To use the agent the following code can be used:
```python
from player import DQNPlayer, LastMovePlayer
player = LastMovePlayer(DQNPlayer(mode='test', path'path/to/agent_model.pth'))
```

The path of the most successful agents are:
 1. `models/model_5_2000K_MIX_1024S.pth`
 2. . . .

## Extra - Human player
The `HumanPlayer` is a player that can be used to play against the agent, it's a simple player that asks the user to insert the action to do and it's used to understand how good is the agent policy.  
