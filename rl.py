from SAC import SAC
from env_gan import GANEnv
import matplotlib.pyplot as plt
import os

gan_model_name='stylegan2_ffhq512'
text_prompt='a photo of a similing face'
threshold=0.5
epsilon=0.025

env = GANEnv(gan_model_name=gan_model_name, text_prompt=text_prompt, threshold=threshold, epsilon=epsilon)

model = SAC('MlpPolicy', env, verbose=1, buffer_size=10)
model.learn(total_timesteps=1)

text_prompt_folder = text_prompt.replace(' ', '_')

experiment_path = os.path.join('rl_experiments', gan_model_name, text_prompt_folder, f"th={threshold}_ep={epsilon}")

model.save(os.path.join(experiment_path, "sac"))

done = False
observation, _ = env.reset(seed=42)
action = env.action_space.sample()
episode = 0

while not done:
    action, _ = model.predict(observation=observation, deterministic=True)
    observation, reward, terminated, truncated, info = env.step(action)
    done = truncated or terminated
    print(f"Reward: {reward}")
    image = info["image"]
    plt.imshow(image)
    plt.show()
    os.makedirs(os.path.join(experiment_path, 'images'), exist_ok=True)
    image.save(os.path.join(experiment_path, 'images', f'ep={episode}.jpg'))
    episode += 1