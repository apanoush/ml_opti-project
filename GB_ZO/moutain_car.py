import gymnasium as gym
import numpy as np
from tqdm.notebook import tqdm


# Note: D=51 => very slow with Multipoint Gradient Estimator, but fast with SPSA.
class MLPPolicy:
    """
    Simple MLP to predict the action to take based on the current state of the car.
    """
    def __init__(self, input_dim=2, hidden_dim=8, output_dim=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = np.random.randn(hidden_dim, input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim)
        self.b2 = np.zeros(output_dim)

    def act(self, obs):
        """
        Chooses the next action.
        :param obs: current state of the car.
        :return: next action.
        """
        z1 = np.tanh(self.W1 @ obs + self.b1)
        out = self.W2 @ z1 + self.b2
        return int(np.argmax(out))

    def get_params(self):
        """
        Returns the flattened array of parameters of the MLP.
        :return: array of parameters.
        """
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        """
        Updates the parameters of the MLP.
        :param params: new values of the parameters
        """
        num_updated = 0  # params is a flattened array, so we keep track of the elements already updated.
        W1_size = self.hidden_dim * self.input_dim
        self.W1 = params[num_updated:num_updated + W1_size].reshape(self.hidden_dim, self.input_dim)
        num_updated += W1_size

        self.b1 = params[num_updated:num_updated + self.hidden_dim]
        num_updated += self.hidden_dim

        W2_size = self.output_dim * self.hidden_dim
        self.W2 = params[num_updated:num_updated + W2_size].reshape(self.output_dim, self.hidden_dim)
        num_updated += W2_size

        self.b2 = params[num_updated:num_updated + self.output_dim]


def compute_step_reward(obs, goal_position):
    """
    Custom reward function to help learning in zeroth-order setting.
    :param obs: current state of the car.
    :param goal_position: position of the objective to reach.
    :return: custom reward for this step.
    """
    position = obs[0]  # [-1.2, 0.6]
    velocity = obs[1]  # [-0.07, 0.07]
    # We give negative reward when the car does not reach the goal,
    # as to not encourage "farming" before reaching the goal.
    # We give a large reward when the car reaches the goal.
    reward = (
            100 * (position >= goal_position) +
            5 * abs(velocity) +                   # [0, 0.35]
            0.2 * (np.sin(3 * position) + 1) -    # [0, 0.4]
            1
    )
    return reward


def evaluate_policy(env, policy, n_episodes=5):
    """
    Evaluate the policy by computing the mean total reward accumulated over a number of episodes.
    :param env: environment (mountain car)
    :param policy: network that chooses the actions.
    :param n_episodes: number of episodes used to evaluate the policy.
    :return: mean total reward accumulated over `n_episodes` episodes
    """
    total_reward = 0.0
    # Repeat for n episodes
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        # Run the episode
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            reward = compute_step_reward(obs, env.unwrapped.goal_position)

            # Accumulate the rewards
            total_reward += reward
            done = terminated or truncated

    # Return the mean total reward
    return total_reward / n_episodes


def train(gradient_function, iterations=10000, alpha_init=0.5, K_init=5.0, alpha_decay=0.01, K_decay=0.01, hidden_dim=8):
    """
    Trains a MLPPolicy to solve the MountainCar RL problem with a zeroth-order gradient approximation function.
    :param gradient_function: gradient estimation function (SPSA or Multipoint Gradient Estimator).
    :param iterations: number of training iterations.
    :param alpha_init: initial learning rate
    :param K_init: initial perturbations magnitude for the gradient estimation.
    :param alpha_decay: learning rate decay (alpha = alpha_init / (1 + alpha_decay x t), t = num. iterations)
    :param K_decay: perturbations magnitude decay (K = K_init / (1 + K_decay x t), t = num. iterations)
    :param hidden_dim: dimension of the hidden layer of MLPPolicy.
    :return: rewards history, and best policy parameters.
    """
    # Set the numpy seed
    np.random.seed(42)
    # Create the environment
    env = gym.make("MountainCar-v0")
    # Set the environment seed
    env.reset(seed=42)

    # Initialize the MLP
    input_dim = env.observation_space.shape[0]  # [2]
    output_dim = env.action_space.n  # [3]
    policy = MLPPolicy(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    # Initialize tracking variables
    best_params = policy.get_params()
    best_reward = -np.inf
    reward_history = []

    # Define the function to maximize w.r.t. the parameters.
    # In our case, we want to maximize the rewards.
    def reward_function(params):
        policy.set_params(params)
        return evaluate_policy(env, policy)

    # Run the training for
    for step in tqdm(range(iterations)):

        # Decay alpha and K
        alpha = alpha_init / (1 + alpha_decay * step)
        K = K_init / (1 + K_decay * step)

        # Estimate the gradient (which takes 5 episodes)
        x = policy.get_params()
        grad = gradient_function(x, reward_function, K)

        # Update the parameters
        # Note: We want to maximize!
        new_params = x + alpha * grad

        # Compute the mean total accumulated reward over 5 episodes and update the history.
        current_reward = reward_function(new_params)
        reward_history.append(current_reward)

        # To help training, only keep the new parameters if the new reward is better than the best reward up until now.
        if current_reward > best_reward:
            tqdm.write(f"Step {step} - New best reward: {current_reward: .2f}")
            best_reward = current_reward
            best_params = new_params.copy()
        # Otherwise, revert to the best policy parameters.
        else:
            policy.set_params(best_params)

    env.close()
    return reward_history, best_params


def visualize_policy(params, hidden_dim=8):
    """
    Renders one episode of the MountainCar environment with the given policy paremeters.
    :param params: parameters of the MLPPolicy, typically the best ones.
    :param hidden_dim: dimension of the hidden layer of MLPPolicy.
    """
    # Create the environment
    env = gym.make("MountainCar-v0", render_mode="human")
    # Set the seed for visualization
    env.reset(seed=0)

    # Initialize the MLPPolicy with the given parameters.
    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy = MLPPolicy(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    policy.set_params(params)

    # Run and render the episode.
    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()
