from SAC import SAC
from env_gan import GANEnv
import matplotlib.pyplot as plt
import os
from custom_callback import CustomCallback
from stable_baselines3.common.vec_env import SubprocVecEnv

import argparse

def parse_args():
	parser = argparse.ArgumentParser(description = 'RL GAN PATHS')

	parser.add_argument('--gan_model_name', type = str, help = 'Name of the model.', default='stylegan2_ffhq512')
	parser.add_argument('--text_prompt', type = str, help = 'Text prompt.', default='Test prompt')
	parser.add_argument('--threshold', type = float, help = 'Threshold of similarity for termination.', default=0.5)
	parser.add_argument('--epsilon', type = float, help = 'Module of the step.', default=0.025)
	parser.add_argument('--batch_size', type = int, help = 'Batch size.', default=2)
	parser.add_argument('--theta', type = int, help = 'Number of the first layers of GAN that will be modified by actions', default=8)
	parser.add_argument('--eta', type = float, help = 'Weight of IDloss reward (reward_id).', default=1)
	parser.add_argument('--save_interval', type = int, help = 'The frequency in which the images are saved.', default=100)
	parser.add_argument('--buffer_size', type = int, help = 'The buffer size of the sac algorithm.', default=1)     
	parser.add_argument('--total_timestep', type = int, help = 'The total timesteps for training.',default=1)
	parser.add_argument('--eval_mode', type = bool, help = 'Run env in eval mode (generate images).',default=False)
              

	args = vars(parser.parse_args())
	args = {k: v for k, v in args.items() if v is not None}

	return args

def make_env(gan_model_name, text_prompt, threshold, epsilon, seed):
    def _init():
        env = GANEnv(gan_model_name=gan_model_name, text_prompt=text_prompt, threshold=threshold, epsilon=epsilon, batch_size=1)
        env.reset(seed)
        return env
    return _init

def train(env,
          save_interval,
          buffer_size,
          total_timestep,
          experiment_path):

    model = SAC('MlpPolicy', env, verbose=1, buffer_size=buffer_size)

    save_image_callback = CustomCallback(save_freq=save_interval, save_path=os.path.join(experiment_path, 'images', 'train'))

    model.learn(total_timesteps=total_timestep, callback=save_image_callback)

    model.save(os.path.join(experiment_path, "sac"))

    return model

def validation(env, model, experiment_path):
    env.eval_mode = True
    done = False
    observation, _ = env.reset(seed=42)
    action = env.action_space.sample()
    episode = 0

    while not done:
        action, _ = model.predict(observation=observation, deterministic=True)
        observation, reward, terminated, truncated, info = env.step(action)
        #observation, reward, done, info = env.step(action)
        #print(f"DONE: {done}")
        done = truncated or terminated
        print(f"Reward: {reward}")
        images = info["image"]
        os.makedirs(os.path.join(experiment_path, 'images', 'val'), exist_ok=True)
        for i, image in enumerate(images):
            image.save(os.path.join(experiment_path, 'images', 'val', f'image_{i}_ep={episode}.jpg'))
        episode += 1
    return     

def main():
    args = parse_args()

    gan_model_name=args['gan_model_name']
    text_prompt=args['text_prompt']
    threshold=args['threshold']
    epsilon=args['epsilon']
    batch_size=args['batch_size']
    theta=args['theta']
    eta=args['eta']
    save_interval=args['save_interval']
    buffer_size=args['buffer_size']
    total_timestep=args['total_timestep']
    eval_mode = args['eval_mode']

    # Some checks
    assert 'stylegan' in gan_model_name

    text_prompt_folder = text_prompt.replace(' ', '_')
    experiment_path = os.path.join('rl_experiments', gan_model_name, text_prompt_folder, f"th={threshold}_ep={epsilon}")

    env = GANEnv(gan_model_name=gan_model_name, 
                 text_prompt=text_prompt, 
                 eval_mode=eval_mode, 
                 threshold=threshold, 
                 epsilon=epsilon, 
                 batch_size=batch_size, 
                 theta=theta, 
                 eta=eta)

    print(f"TRAINING...")
    model = train(env,
          save_interval,
          buffer_size,
          total_timestep,
          experiment_path)
    print(" DONE")
    
    if eval_mode:
         validation(env, model, experiment_path)

if __name__ == '__main__':
     main()