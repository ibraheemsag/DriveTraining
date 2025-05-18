import gymnasium as gym
import numpy as np
import itertools
import pickle
from collections import deque
import argparse
from pytux_gym import Reward2Wrapper, AimpointRewardWrapper, FullControlWrapper, SteerOnlyWrapper
from tqdm import tqdm
import time
# ---------------------------------------------------------------------------
# Parse command-line flags
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Tabular Q-learning on SuperTuxKart with discrete wrapper")
parser.add_argument("--penalty", type=float, default=0.1, help="Penalty factor for Reward2Wrapper")
parser.add_argument("--align", type=float, default=0.05, help="Alignment factor for AimpointRewardWrapper")
parser.add_argument("--speed_reward", type=float, default=0.05, help="Base speed reward coefficient (env parameter)")
parser.add_argument("--distance_reward", type=float, default=0.1, help="Base distance-progress reward coefficient (env parameter)")
parser.add_argument("--episodes", type=int, default=500, help="Number of training episodes")
parser.add_argument("--time_penalty", type=float, default=3, help="Time penalty for Reward2Wrapper")
parser.add_argument("--exp_name", type=str, default="exp", help="Experiment name prefix for saved Q-table")
parser.add_argument("--wrapper", choices=["discrete", "steer"], default="discrete", help="Which action wrapper to use: 'discrete' (DiscreteActionObservationWrapper) or 'steer' (SteerOnlyWrapper)")
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 1.  Build the wrapped *discrete* environment
# ---------------------------------------------------------------------------
base_env = gym.make(
    "PyTuxGym-v0",
    image_only=False,
    render_mode="rgb_array",
    max_steps=1500,
    speed_reward=args.speed_reward,
    distance_reward=args.distance_reward,
    time_penalty=args.time_penalty
)

# Optional additional reward shaping
env = Reward2Wrapper(base_env, penalty=args.penalty)
env = AimpointRewardWrapper(env, alignment_factor=args.align)

# Choose action/observation wrapper
if args.wrapper == "discrete":
    env = FullControlWrapper(env)
else:
    env = SteerOnlyWrapper(env)

# Debug spaces to confirm
print("Obs space :", env.observation_space)
print("Act space :", env.action_space)

# ---------------------------------------------------------------------------
# 2.  Helper functions to flatten / unflatten state-action indices
# ---------------------------------------------------------------------------
state_bins   = env.observation_space.nvec          # [4, 3]
action_bins  = env.action_space.nvec               # [5, 2]
N_S, N_A     = np.prod(state_bins), np.prod(action_bins)

def flat_state(s):
    """Flatten discrete observation vector to a unique integer state index."""
    return np.ravel_multi_index(s, state_bins)

def unflat_act(a_idx):  # returns the 2-dim action expected by env.step()
    return np.array(np.unravel_index(a_idx, action_bins), dtype=np.int64)

# ---------------------------------------------------------------------------
# 3.  Initialise Q-table and learning hyper-parameters
# ---------------------------------------------------------------------------
Q           = np.zeros((N_S, N_A), dtype=np.float32)
n_episodes  = args.episodes
gamma       = 0.99
alpha       = 0.10           # learning-rate
eps_start   = 1.0            # ε-greedy schedule
eps_end     = 0.05
eps_decay   = 0.999          # ε ← ε·decay each episode

reward_log  = deque(maxlen=100)
pbar = tqdm(range(1, n_episodes + 1), desc="Training", unit="ep")
# ---------------------------------------------------------------------------
# 4.  Training loop
# ---------------------------------------------------------------------------
eps = eps_start
for ep in pbar:
    obs, _ = env.reset(seed=ep)
    s_idx  = flat_state(obs)
    done   = truncated = False
    total_reward = 0.0

    while not (done or truncated):
        # ε-greedy action selection on flattened space
        if np.random.random() < eps:
            a_idx = np.random.randint(N_A)
        else:
            a_idx = Q[s_idx].argmax()

        next_obs, r, done, truncated, _ = env.step(unflat_act(a_idx))
        next_s_idx = flat_state(next_obs)

        # Q-learning update
        best_next = Q[next_s_idx].max()
        td_error  = r + gamma * best_next - Q[s_idx, a_idx]
        Q[s_idx, a_idx] += alpha * td_error

        s_idx  = next_s_idx
        total_reward += r

    # book-keeping
    reward_log.append(total_reward)
    eps = max(eps_end, eps * eps_decay)

    if ep % 100 == 0:
        print(f"Episode {ep:4d} | ε={eps:.3f} | "
              f"Score(avg100)={np.mean(reward_log):8.2f}")

# ---------------------------------------------------------------------------
# 5.  Save the learned table with timestamp
# ---------------------------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
fname = (
    f"{args.exp_name}_pen{args.penalty}_a{args.align}_sr{args.speed_reward}_dr{args.distance_reward}_tp{args.time_penalty}_ep{args.episodes}_{timestamp}.pkl"
)
with open(fname, "wb") as f:
    pickle.dump(Q, f)
print(f"Training complete – Q-table saved to '{fname}'")

env.close()
