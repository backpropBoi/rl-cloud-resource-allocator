# src/train.py
import os
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from src.envs.cloud_env import CloudEnv
from src.callbacks import TrainingLoggingCallback

def make_env():
    env = CloudEnv(max_workers=20, arrival_rate=3.0)
    return Monitor(env)

def main():
    model_dir = os.environ.get('MODEL_DIR', './models')
    os.makedirs(model_dir, exist_ok=True)

    logs_dir = os.environ.get('LOG_DIR', './logs')
    os.makedirs(logs_dir, exist_ok=True)

    env = DummyVecEnv([make_env])

    model = SAC('MlpPolicy',
                env,
                verbose=1,
                learning_rate=3e-4,
                buffer_size=200_000,
                batch_size=256,
                learning_starts=1000,
                ent_coef='auto')

    checkpoint_callback = CheckpointCallback(save_freq=50_000, save_path=model_dir, name_prefix='sac_cloud')
    log_callback = TrainingLoggingCallback(log_dir=logs_dir, log_file="training_log.csv")

    timesteps = 20_000
    # provide a list of callbacks; SB3 will call each
    model.learn(total_timesteps=timesteps, callback=[checkpoint_callback, log_callback])
    model.save(os.path.join(model_dir, 'sac_cloud_final'))

if __name__ == '__main__':
    main()
