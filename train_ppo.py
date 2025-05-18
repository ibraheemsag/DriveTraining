"""
Train a PPO agent to control the kart using the trained planner for aimpoints.

This script uses Stable Baselines 3 to train a PPO agent on the PyTuxGymSimple-v0
environment, which uses the simplified action space (acceleration and steering only)
and relies on the pre-trained planner for aimpoints. This version uses Stable Baselines 3's
built-in feature extractors rather than a custom one.
"""
import os
import time
import json
import numpy as np
import gymnasium as gym
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import argparse
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed

# Import our environment (this makes sure it's registered)
from pytux_gym import PyTuxEnv, SimpleActionWrapper, Reward2Wrapper, AimpointRewardWrapper


def parse_args():
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(description='Train a PPO agent for SuperTuxKart racing')
    
    parser.add_argument('--experiment_name', type=str, default='simple_extractor',
                        help='Name of the experiment for organizing runs')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total timesteps to train for')
    parser.add_argument('--use_reward2', type=bool, default=True,
                        help='Use the Reward2Wrapper to penalize backward movement')
    parser.add_argument('--use_aimpoint_reward', type=bool, default=False,
                        help='Use the AimpointRewardWrapper to incentivize following the aimpoint')
    parser.add_argument('--aimpoint_factor', type=float, default=0.1,
                        help='Reward factor for the AimpointRewardWrapper')
    parser.add_argument('--penalty', type=float, default=0.1,
                        help='Penalty factor for the Reward2Wrapper')
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    SEED = args.seed
    set_random_seed(SEED)

    # Track selection - Focus on La Hacienda
    MAIN_TRACK = 'hacienda'
    TRAIN_TRACKS = [MAIN_TRACK]  # Single track for consistent training
    EVAL_TRACKS = ['hacienda', 'zengarden']  # Evaluate on both the training track and a different one

    # Create directory structure for logs and models
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join("runs", args.experiment_name)
    run_name = f"run_{timestamp}"
    run_dir = os.path.join(experiment_dir, run_name)
    model_dir = os.path.join(run_dir, "models")
    
    # Create directories
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    # Create a TensorBoard writer
    tb_writer = SummaryWriter(log_dir=run_dir)

    # Save experiment config
    experiment_config = {
        "experiment_name": args.experiment_name,
        "timestamp": timestamp,
        "seed": SEED,
        "train_tracks": TRAIN_TRACKS,
        "eval_tracks": EVAL_TRACKS,
        "total_timesteps": args.timesteps,
        "use_reward2": args.use_reward2,
        "use_aimpoint_reward": args.use_aimpoint_reward,
        "aimpoint_factor": args.aimpoint_factor,
        "penalty": args.penalty,
        "ppo_params": {
            "learning_rate": 3e-4,
            "n_steps": 2048,
            "batch_size": 64,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5
        }
    }

    # Log configuration to TensorBoard as text
    config_str = json.dumps(experiment_config, indent=2)
    tb_writer.add_text("experiment_config", config_str, 0)

    # Save the configuration
    with open(os.path.join(run_dir, "experiment_config.json"), 'w') as f:
        json.dump(experiment_config, f, indent=2)

    def make_env(tracks=TRAIN_TRACKS, seed=0, log_dir=None, idx=0, monitor=True, 
                 use_reward2=args.use_reward2, 
                 use_aimpoint_reward=args.use_aimpoint_reward,
                 aimpoint_factor=args.aimpoint_factor,
                 penalty=args.penalty):
        """
        Create a wrapped and monitored environment.
        """
        def _init():
            # Create the base environment with planner aimpoints
            env = gym.make("PyTuxGymSimple-v0", 
                        tracks=tracks,
                        render_mode="rgb_array",
                        image_only=False,
                        use_planner_aimpoint=True)
            
            # Optionally apply Reward2Wrapper
            if use_reward2:
                env = Reward2Wrapper(env, penalty=penalty)
            
            # Optionally apply AimpointRewardWrapper
            if use_aimpoint_reward:
                env = AimpointRewardWrapper(env, alignment_factor=aimpoint_factor)
            
            # Set the seed
            env.reset(seed=seed)
            return env
        
        return _init

    # Create vectorized training environment with monitoring
    env = DummyVecEnv([make_env(seed=SEED)])
    env = VecMonitor(env, os.path.join(run_dir, "train_monitor"))
    
    # Create separate evaluation environment
    eval_env = DummyVecEnv([make_env(tracks=EVAL_TRACKS, seed=SEED+1)])
    eval_env = VecMonitor(eval_env, os.path.join(run_dir, "eval_monitor"))
    
    # We need to transpose the image for CNN processing
    env = VecTransposeImage(env)
    eval_env = VecTransposeImage(eval_env)

    # Configure the policy parameters using default extractors
    policy_kwargs = dict(
        net_arch=dict(pi=[128, 64], vf=[128, 64]),
    )

    # Log model architecture to TensorBoard
    if tb_writer:
        tb_writer.add_text("model/architecture", str(policy_kwargs), 0)

    # Configure the PPO agent
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        normalize_advantage=True,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        target_kl=None,
        tensorboard_log=run_dir,
        verbose=1,
        seed=SEED,
        device="auto",
    )

    # Checkpoint callback - save models periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=model_dir,
        name_prefix="ppo_simple",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )
    
    # Evaluation callback - evaluate on separate environment periodically
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(model_dir, "best_model"),
        log_path=os.path.join(run_dir, "eval_results"),
        eval_freq=10000,
        deterministic=True,
        render=False,
        n_eval_episodes=5,
        verbose=1
    )

    # Combine callbacks
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    # Train the agent
    total_timesteps = args.timesteps
    print(f"\n{'='*80}\nStarting training for experiment '{args.experiment_name}' (run_{timestamp})")
    print(f"Training on {MAIN_TRACK} for {total_timesteps} timesteps...")
    print(f"Logs and models will be saved to: {run_dir}")
    print(f"Training track: {MAIN_TRACK}")
    print(f"Evaluation tracks: {EVAL_TRACKS}")
    print(f"TensorBoard: Run 'tensorboard --logdir {experiment_dir}' to see all runs in this experiment")
    print(f"TensorBoard URL: http://localhost:6006")
    print(f"{'='*80}\n")

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        tb_log_name="ppo",
    )

    # Close environments
    env.close()
    eval_env.close()

    # Save the final model
    final_model_path = os.path.join(model_dir, "final_model")
    model.save(final_model_path)
    print(f"\n{'='*80}\nTraining complete!\nFinal model saved to {final_model_path}")
    print(f"View metrics in TensorBoard: tensorboard --logdir {experiment_dir}")
    print(f"TensorBoard URL: http://localhost:6006")
    print(f"{'='*80}\n")

    return True


if __name__ == "__main__":
    main() 