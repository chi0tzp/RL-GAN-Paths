from SAC import SAC
from env_gan import GANEnv
import matplotlib.pyplot as plt
import os
from custom_callback import CustomCallback

gan_model_name='stylegan2_ffhq512'
text_prompt='a photo of a similing face'
threshold=0.5
epsilon=0.025
batch_size = 2

save_interval = 5  # Save image every 1000 steps
text_prompt_folder = text_prompt.replace(' ', '_')
experiment_path = os.path.join('rl_experiments', gan_model_name, text_prompt_folder, f"th={threshold}_ep={epsilon}")

env = GANEnv(gan_model_name=gan_model_name, text_prompt=text_prompt, threshold=threshold, epsilon=epsilon, batch_size=batch_size)

model = SAC('MlpPolicy', env, verbose=1, buffer_size=10)

save_image_callback = CustomCallback(save_freq=save_interval, save_path=os.path.join(experiment_path, 'images', 'train'))

model.learn(total_timesteps=10, callback=save_image_callback)

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
    images = info["image"]
    os.makedirs(os.path.join(experiment_path, 'images', 'val'), exist_ok=True)
    for i, image in enumerate(images):
        image.save(os.path.join(experiment_path, 'images', 'val', f'image_{i}_ep={episode}.jpg'))
    episode += 1