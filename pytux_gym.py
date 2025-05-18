import gymnasium as gym
import numpy as np
from gymnasium import spaces
from gymnasium.envs.registration import register
import torch
import torchvision.transforms.functional as TF
import random
import pystk
from utils import PyTux, TRACK_OFFSET
from planner import load_model
import gc

# List of available tracks in SuperTuxKart
AVAILABLE_TRACKS = [
    'cocoa_temple','cornfield_crossing', 'hacienda', 'lighthouse', 'scotland', 'snowtuxpeak', 'zengarden'
]

class PyTuxEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
    }
    
    # Shared PyTux instance across all environments
    _shared_pytux = None

    def __init__(self, tracks=None, screen_width=128, screen_height=96, 
                 max_steps=1000, render_mode="rgb_array", image_only=False,
                 use_planner_aimpoint=True, terminal_reward=1000, speed_reward=0.05, distance_reward=0.1, time_penalty=0.1):
        super(PyTuxEnv, self).__init__()
        
        # Handle tracks parameter
        if tracks is None:
            # Use all available tracks
            self.tracks = AVAILABLE_TRACKS
        elif isinstance(tracks, str):
            # Single track provided
            self.tracks = [tracks]
        else:
            # List of tracks provided
            self.tracks = tracks
        
        self.current_track = self.tracks[0]  # Will be updated in reset()
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.image_only = image_only
        self.terminal_reward = terminal_reward
        self.s = speed_reward
        self.d = distance_reward
        self.last_distance = 0
        self.time_penalty = time_penalty
        # Whether to use the trained planner for aimpoints
        self.use_planner_aimpoint = use_planner_aimpoint
        self.planner = None
        if use_planner_aimpoint:
            try:
                print("Loading trained planner model...")
                self.planner = load_model().eval()
                print("Planner model loaded successfully!")
            except Exception as e:
                print(f"Error loading planner model: {e}")
                print("Falling back to default aimpoint calculation.")
                self.use_planner_aimpoint = False
        
        self.pytux = None
        
        # Define action space (acceleration, steering, brake, drift)
        # acceleration: [0, 1], steering: [-1, 1], brake: [0, 1], drift: [0, 1]
        self.action_space = spaces.Box(
            low=np.array([0, -1, 0, 0]), 
            high=np.array([1, 1, 1, 1]),
            dtype=np.float32
        )
        
        # Define observation space
        if image_only:
            # Image-only observation space
            self.observation_space = spaces.Box(
                low=0, high=255, 
                shape=(screen_height, screen_width, 3),
                dtype=np.uint8
            )
        else:
            # Flattened combined image and state observation space
            self.observation_space = spaces.Dict({
                # RGB image
                'image': spaces.Box(
                    low=0, high=255, 
                    shape=(screen_height, screen_width, 3),
                    dtype=np.uint8
                ),
                # Velocity (3D vector)
                'velocity': spaces.Box(
                    low=-np.inf, high=np.inf, 
                    shape=(3,), dtype=np.float32
                ),
                # Speed (scalar)
                'speed': spaces.Box(
                    low=0, high=np.inf, 
                    shape=(1,), dtype=np.float32
                ),
                # Track progress (0 to 1)
                'progress': spaces.Box(
                    low=0, high=1, 
                    shape=(1,), dtype=np.float32
                ),
                # Kart forward direction
                'front': spaces.Box(
                    low=-1, high=1, 
                    shape=(3,), dtype=np.float32
                ),
                # Aim point (from planner or track data)
                'aimpoint': spaces.Box(
                    low=-1, high=1, 
                    shape=(2,), dtype=np.float32
                )
            })
        
        self.steps = 0
        self.state = None
        self.track_obj = None
        self.last_rescue = 0
        self.current_image = None
        self.current_aimpoint = None
        
    def _get_state_observation(self):
        if self.state is None or self.track_obj is None:
            # Default values if state not initialized
            return {
                'velocity': np.zeros(3, dtype=np.float32),
                'speed': np.array([0], dtype=np.float32),
                'progress': np.array([0], dtype=np.float32),
                'front': np.array([0, 0, 1], dtype=np.float32),
            }
            
        kart = self.state.players[0].kart
        
        # Calculate kart speed
        velocity = np.array(kart.velocity, dtype=np.float32)
        speed = np.array([np.linalg.norm(velocity)], dtype=np.float32)
        
        # Calculate progress and ensure it's in [0, 1]
        progress_value = kart.overall_distance / max(1e-5, self.track_obj.length)
        progress = np.array([np.clip(progress_value, 0, 1)], dtype=np.float32)
        
        # Get kart orientation (front is the forward-facing direction vector)
        # Normalize to ensure it's within [-1, 1] for all components
        front = np.array(kart.front, dtype=np.float32)
        # Normalize the front vector to unit length to ensure it's within [-1, 1]
        front_norm = np.linalg.norm(front)
        if front_norm > 0:
            front = front / front_norm
        
        return {
            'velocity': velocity,
            'speed': speed, 
            'progress': progress,
            'front': front,
        }
        
    def get_aimpoint(self):
        if self.use_planner_aimpoint and self.planner is not None and self.current_image is not None:
            # Use the trained planner to predict the aim point
            # spatial_argmax in the planner already returns normalized [-1, 1] coordinates
            with torch.no_grad():
                image_tensor = TF.to_tensor(self.current_image)[None]  # Add batch dimension
                aim_point = self.planner(image_tensor).squeeze(0).cpu().numpy()
            # Ensure values are strictly within [-1, 1] to avoid warnings
            return aim_point
        else:
            # Fallback: return the centre of the image (0,0) in normalised coordinates
            print("Using default aimpoint calculation.")
            return np.zeros(2, dtype=np.float32)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Get random state for reproducibility
        rng = random.Random(seed)
        
        # Select track - either from options or randomly
        if options and 'track' in options:
            self.current_track = options['track']
        else:
            self.current_track = rng.choice(self.tracks)
        
        # Initialize PyTux as a singleton instance
        if PyTuxEnv._shared_pytux is None:
            print("Creating shared PyTux instance...")
            PyTuxEnv._shared_pytux = PyTux(self.screen_width, self.screen_height, force_singleton=True, render_mode=self.render_mode)
        
        # Use the shared instance
        self.pytux = PyTuxEnv._shared_pytux
        
        # Reset internal state
        self.steps = 0
        self.last_rescue = 0
        self.last_distance = 0  # Initialize last_distance to 0
        
        # Initialize the race
        if self.pytux.k is not None and self.pytux.k.config.track == self.current_track:
            self.pytux.k.restart()
            self.pytux.k.step()
        else:
            if self.pytux.k is not None:
                self.pytux.k.stop()
                del self.pytux.k
                
            config = pystk.RaceConfig(num_kart=1, laps=1, track=self.current_track)
            config.players[0].controller = pystk.PlayerConfig.Controller.PLAYER_CONTROL
            
            self.pytux.k = pystk.Race(config)
            self.pytux.k.start()
            self.pytux.k.step()
        
        # Create state and track objects
        self.state = pystk.WorldState()
        self.track_obj = pystk.Track()
        
        self.state.update()
        self.track_obj.update()
        
        # Get initial observation
        if self.render_mode == "rgb_array":
            image = np.array(self.pytux.k.render_data[0].image, dtype=np.uint8)
            self.current_image = image
        
        # Get aim point
        self.current_aimpoint = self.get_aimpoint()
        
        # Get state observations for debugging
        state_obs = self._get_state_observation()
        
        # Debug print to find out of bounds values
        if not self.image_only:
            # Check image bounds
            image_shape = image.shape
            image_min, image_max = np.min(image), np.max(image)
            if image_min < 0 or image_max > 255:
                print(f"DEBUG: Image values out of bounds: min={image_min}, max={image_max}")
            
            # Check velocity bounds (should be fine as -inf to inf)
            
            # Check speed - should be non-negative
            speed = state_obs['speed']
            if np.any(speed < 0):
                print(f"DEBUG: Speed negative: {speed}")
            
            # Check progress - should be in [0, 1]
            progress = state_obs['progress']
            if np.any(progress < 0) or np.any(progress > 1):
                print(f"DEBUG: Progress out of bounds: {progress}")
            
            # Check front vector - should be in [-1, 1]
            front = state_obs['front']
            if np.any(np.abs(front) > 1):
                print(f"DEBUG: Front vector out of bounds: {front}")
            
            # Check aimpoint - should be in [-1, 1]
            aimpoint = self.current_aimpoint
            if np.any(np.abs(aimpoint) > 1):
                print(f"DEBUG: Aimpoint out of bounds: {aimpoint}")
        
        # Construct observation based on configuration
        if self.image_only:
            observation = image
        else:
            observation = {
                'image': image,
                'velocity': state_obs['velocity'],
                'speed': state_obs['speed'],
                'progress': state_obs['progress'],
                'front': state_obs['front'],
                'aimpoint': self.current_aimpoint
            }
        
        info = {
            'track': self.current_track,
            'aimpoint': self.current_aimpoint
        }
        return observation, info
    
    def step(self, action):
        self.steps += 1
        
        # Update state
        self.state.update()
        self.track_obj.update()
        
        kart = self.state.players[0].kart
        rescue = False
        # Create action object
        pystk_action = pystk.Action()
        pystk_action.acceleration = float(action[0])
        pystk_action.steer = float(action[1])
        pystk_action.brake = float(action[2])
        pystk_action.drift = bool(action[3] > 0.5)
        
        # Handle rescues (if kart is stuck)
        current_vel = np.linalg.norm(kart.velocity)
        if current_vel < 1.0 and self.steps - self.last_rescue > 30:  # RESCUE_TIMEOUT
            self.last_rescue = self.steps
            pystk_action.rescue = True
            rescue = True
        
        # Apply action and get new observation
        self.pytux.k.step(pystk_action)
        if self.render_mode == "rgb_array":
            image = np.array(self.pytux.k.render_data[0].image, dtype=np.uint8)
            self.current_image = image
        
        # Get aim point
        self.current_aimpoint = self.get_aimpoint()
        
        # Construct observation based on configuration
        if self.image_only:
            observation = image
        else:
            observation = {
                'image': image,
                'velocity': self._get_state_observation()['velocity'],
                'speed': self._get_state_observation()['speed'],
                'progress': self._get_state_observation()['progress'],
                'front': self._get_state_observation()['front'],
                'aimpoint': self.current_aimpoint
            }
        
        # Calculate distance traversed on track
        progress = kart.overall_distance / self.track_obj.length
        
        # Determine if race is complete
        terminated = False
        if np.isclose(progress, 1.0, atol=2e-3):
            terminated = True
        
        # Determine if episode is truncated (e.g., max steps)
        truncated = self.steps >= self.max_steps
        d_reward = self.d * (kart.overall_distance - self.last_distance)
        self.last_distance = kart.overall_distance
        speed_reward = self.s * np.linalg.norm(kart.velocity)
        # Calculate reward (example reward function)
        ### TODO: Include signals to listen to the controller panel
        ### TODO: Consider adding a penalty for sudden shifting
        ### TODO: Consider adding reward for speed along the track 
        ### TODO: Add large bonus for terminating the race
        reward = d_reward + speed_reward - 0.1 * self.time_penalty

        # Bonus for completing the track
        if terminated:
            reward += self.terminal_reward
        
        # Provide additional info
        info = {
            'track': self.current_track,
            'progress': progress,
            'velocity': current_vel,
            'distance': kart.overall_distance,
            'steps': self.steps,
            'aimpoint': self.current_aimpoint,
            'true_velocity': kart.velocity,
            'rescue': rescue
        }
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        if self.render_mode == "rgb_array" and self.current_image is not None:
            return self.current_image
        return None
    
    def close(self):
        if hasattr(self, '_viewer') and self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            
        if PyTuxEnv._shared_pytux is not None and self.pytux is PyTuxEnv._shared_pytux:

            if self.pytux.k is not None:
                self.pytux.k.stop()
                

        gc.collect()


class SimpleActionWrapper(gym.Wrapper):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, env=None, **kwargs):
        # If env is None or a string, create a PyTuxEnv environment
        if env is None or isinstance(env, str):
            env = gym.make("PyTuxGym-v0", **kwargs)
            
        super(SimpleActionWrapper, self).__init__(env)
        
        # Define simplified action space (only acceleration and steering)
        # acceleration: [0, 1], steering: [-1, 1]
        self.action_space = spaces.Box(
            low=np.array([0, -1]), 
            high=np.array([1, 1]),
            dtype=np.float32
        )
    
    def step(self, action):
        # Convert 2D action to 4D action (add brake=0 and drift=0)
        full_action = np.zeros(4, dtype=np.float32)
        full_action[0] = action[0]  # acceleration
        full_action[1] = action[1]  # steering
        full_action[2] = 0.0        # brake
        full_action[3] = 0.0        # drift
        
        # Call the base environment's step method with the full action
        return self.env.step(full_action)

# Register environments with gymnasium

class Reward2Wrapper(gym.Wrapper):
    def __init__(self, env, penalty=0.1):
        super().__init__(env)
        self.penalty = penalty
        self.last_distance = 0
        self.first_step = True

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        distance = info['distance']
        
        # Skip penalty on first step, only update reference distance
        if not self.first_step:
            distance_change = distance - self.last_distance
            if info['rescue']:
                # Large backwards movement (exploiting before start line): heavy penalty
                if distance_change <= -100:
                    reward -= self.penalty * abs(distance_change) * 50  # Double penalty for major cheating
                
                # Smaller backwards movement (including respawns): normal penalty
                else:
                    reward -= self.penalty * abs(distance_change)
        else:
            self.first_step = False
        # Update last distance for next step
        self.last_distance = distance
        
        return obs, reward, terminated, truncated, info


class AimpointRewardWrapper(gym.Wrapper):
    def __init__(self, env, alignment_factor=0.1):
        super().__init__(env)
        self.alignment_factor = alignment_factor
    
    def step(self, action):
        # Get the next state from the environment
        obs, reward, terminated, truncated, info = super().step(action)        
        aimpoint = obs['aimpoint']
        aimpoint_centered_reward = 1.0 - abs(aimpoint[0])                
        reward += self.alignment_factor * aimpoint_centered_reward        
        return obs, reward, terminated, truncated, info


class ObservationDiscrete(gym.Wrapper):
    _STEER_VALS  = np.array([-1, -0.5, 0, 0.5, 1], np.float32)  # keep for reuse
    _CENTER_CENTRES = np.linspace(-0.75, 0.75, 4, dtype=np.float32)
    _SPEED_CENTRES  = np.array([6, 18, 30], dtype=np.float32)    # bins: 0-12, 12-24, 24-36

    def __init__(self, env: gym.Env):
        super().__init__(env)
        # every child wrapper will override action_space, but they all share this:
        self.observation_space = spaces.MultiDiscrete(
            [len(self._CENTER_CENTRES), len(self._SPEED_CENTRES)]
        )

    @staticmethod
    def _nearest_bin(val, centres):
        return int(np.abs(centres - val).argmin())

    # snap the dict obs coming from the base env
    def observation(self, observation):
        centre_val = float(observation['aimpoint'][0])
        speed_val  = float(observation['speed'][0])
        return np.array(
            [
                self._nearest_bin(centre_val, self._CENTER_CENTRES),
                self._nearest_bin(speed_val,  self._SPEED_CENTRES),
            ],
            dtype=np.int64,
        )

    # make sure reset/step pass through observation()
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        raise NotImplementedError   # implemented by concrete subclasses


# ---------------------------------------------------------------------
# 2)  Full-control wrapper  (steer + throttle) – old behaviour
# ---------------------------------------------------------------------
class FullControlWrapper(ObservationDiscrete):
    _ACCEL_VALS = np.array([0.0, 1.0], np.float32)   # tap or full

    def __init__(self, env):
        super().__init__(env)
        self.action_space = spaces.MultiDiscrete([5, 2])           # steer, accel
        self._tmpl = np.zeros_like(self.env.action_space.low)
        # index meaning in STK Box: [accel, steer, brake, drift]

    def _map_action(self, discrete):
        steer_bin, accel_bin = map(int, discrete)
        act = self._tmpl.copy()
        act[0] = self._ACCEL_VALS[accel_bin]        # accel
        act[1] = self._STEER_VALS[steer_bin]        # steer
        return act

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(self._map_action(action))
        return self.observation(obs), r, term, trunc, info


# ---------------------------------------------------------------------
# 3)  Fixed-throttle wrapper  (steer + drift) – new request
# ---------------------------------------------------------------------
class SteerOnlyWrapper(ObservationDiscrete):
    _DRIFT_VALS = np.array([0.0, 1.0], np.float32)

    def __init__(self, env, throttle=0.8):
        super().__init__(env)
        self.throttle = float(throttle)
        self.action_space = spaces.MultiDiscrete([5, 2])           # steer, drift
        self._tmpl = np.zeros_like(self.env.action_space.low)
        self._tmpl[0] = self.throttle                              # accel fixed
        # brake (idx 2) left 0

    def _map_action(self, discrete):
        steer_bin, drift_bin = map(int, discrete)
        act = self._tmpl.copy()
        act[1] = self._STEER_VALS[steer_bin]        # steer
        act[3] = self._DRIFT_VALS[drift_bin]        # drift
        return act

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(self._map_action(action))
        return self.observation(obs), r, term, trunc, info
try:
    register(
        id="PyTuxGym-v0",
        entry_point=PyTuxEnv,
        max_episode_steps=1000,
    )
except Exception as e:
    print(f"Note: {e}")

try:
    register(
        id="PyTuxGymSimple-v0",
        entry_point=SimpleActionWrapper,
        max_episode_steps=1000,
    )
except Exception as e:
    print(f"Note: {e}")


