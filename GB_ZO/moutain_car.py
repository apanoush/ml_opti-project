import gymnasium as gym
import numpy as np
from tqdm import tqdm


# Note: D=99 => very slow with multi, fast with SPSA
class MLPPolicy:
    def __init__(self, input_dim=2, hidden_dim=16, output_dim=3):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Initialize weights
        self.W1 = np.random.randn(hidden_dim, input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(output_dim, hidden_dim)
        self.b2 = np.zeros(output_dim)

    def act(self, obs):
        z1 = np.tanh(self.W1 @ obs + self.b1)
        out = self.W2 @ z1 + self.b2
        return int(np.argmax(out))

    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def set_params(self, params):
        offset = 0
        W1_size = self.hidden_dim * self.input_dim
        self.W1 = params[offset:offset + W1_size].reshape(self.hidden_dim, self.input_dim)
        offset += W1_size

        self.b1 = params[offset:offset + self.hidden_dim]
        offset += self.hidden_dim

        W2_size = self.output_dim * self.hidden_dim
        self.W2 = params[offset:offset + W2_size].reshape(self.output_dim, self.hidden_dim)
        offset += W2_size

        self.b2 = params[offset:offset + self.output_dim]


def compute_step_reward(obs, goal_position):
    position = obs[0]
    velocity = obs[1]
    reward = (
            100 * (position >= goal_position) +
            5 * abs(velocity) +
            (np.sin(3 * position) + 1) / 3 -
            1
    )
    return reward


def evaluate_policy(env, policy, n_episodes=5):
    total_reward = 0.0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = policy.act(obs)
            obs, _, terminated, truncated, _ = env.step(action)

            reward = compute_step_reward(obs, env.unwrapped.goal_position)

            total_reward += reward
            done = terminated or truncated

    return total_reward / n_episodes


def train(gradient_function, steps=10000, alpha_init=0.5, K_init=0.05, hidden_dim=16):
    # Set the numpy seed
    np.random.seed(42)

    env = gym.make("MountainCar-v0")
    # Set the environment seed
    env.reset(seed=42)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy = MLPPolicy(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

    best_params = policy.get_params()
    best_reward = -np.inf
    reward_history = []

    for step in tqdm(range(steps)):
        alpha = alpha_init / (1 + 0.01 * step)
        K = K_init / (1 + 0.01 * step)

        # Function to maximize w.r.t. the parameters
        def reward_function(params):
            policy.set_params(params)
            return evaluate_policy(env, policy)

        x = policy.get_params()
        grad = gradient_function(x, reward_function, K)

        new_params = x + alpha * grad  # We want to maximize

        current_reward = reward_function(new_params)
        reward_history.append(current_reward)

        if current_reward > best_reward:
            print(f"Step {step} - New best reward: {current_reward: .2f}")
            best_reward = current_reward
            best_params = new_params.copy()
        else:
            policy.set_params(best_params)

    env.close()
    return reward_history, best_params


def visualize_policy(params, hidden_dim=16):
    env = gym.make("MountainCar-v0", render_mode="human")
    # Set the seed for visualization
    env.reset(seed=0)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.n
    policy = MLPPolicy(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    policy.set_params(params)

    obs, _ = env.reset()
    done = False
    while not done:
        action = policy.act(obs)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        env.render()
    env.close()




