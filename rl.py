from stable_baselines3 import SAC
from env_gan import GANEnv
import matplotlib.pyplot as plt
import os

gan_model_name='stylegan2_ffhq512'
text_prompt='a person with blue eyes'
threshold=0.95
epsilon=0.15

env = GANEnv(gan_model_name=gan_model_name, text_prompt=text_prompt, threshold=threshold, epsilon=epsilon)

model = SAC('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=10000)

text_prompt_folder = text_prompt.replace(' ', '_')

model.save(os.path.join('rl_experiments', gan_model_name, text_prompt_folder, f"th={threshold}_ep={epsilon}", "sac"))

done = False
observation, _ = env.reset(seed=42)
action = env.action_space.sample()

while not done:
    action, _ = model.predict(observation=observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = truncated or terminated
    print(f"Reward: {reward}")
    image = info["image"]
    
    plt.show()