import os
import torch
import numpy as np
import argparse
import matplotlib.pyplot as plt
from sac import SAC
from env_gan import GANEnv

def evaluate_model(env, agent, num_episodes=5, save_dir="evaluation_results"):
    os.makedirs(save_dir, exist_ok=True)
    
    for i_episode in range(num_episodes):
        state = env.reset()
        done = False
        episode_images = []
        
        while not done:
            action = agent.select_action(state, evaluate=True)
            next_state, reward, reward_clip, reward_id, terminated, info = env.step(action, log_img=True)
            truncated = True if i_episode > 50 else False
            done = terminated or truncated

            # Save the generated image
            if "image" in info:
                episode_images.append(info["image"])

            state = next_state
        
        # Save episode images
        for idx, img in enumerate(episode_images):
            img_save_path = os.path.join(save_dir, f"episode_{i_episode + 1}_step_{idx + 1}.png")
            plt.imsave(img_save_path, img)
        print(f"Episode {i_episode + 1} images saved to {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate SAC Agent')
    # SAC args
    parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', action='store_true',
                    help='Evaluates a policy every 10 episodes (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', action='store_true',
                    help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=1, metavar='N',
                    help='random seed (default: 1)')
    parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='batch size (default: 1)')
    parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                    help='Steps sampling random actions (default: 5000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=1, metavar='N',
                    help='size of replay buffer (default: 1)')
    parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')

    parser.add_argument('--gan_model_name', type=str, help='Name of the GAN model.', default='stylegan2_ffhq512')
    parser.add_argument('--text_prompt', type=str, help='Text prompt.', default='Test prompt')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Path to the model checkpoint.')
    parser.add_argument('--save_dir', type=str, default='evaluation_results', help='Directory to save the generated images.')
    parser.add_argument('--num_episodes', type=int, default=5, help='Number of episodes to evaluate.')

    args = parser.parse_args()

    # Load environment
    env = GANEnv(gan_model_name=args.gan_model_name, 
                 text_prompt=args.text_prompt, 
                 threshold=0.5, 
                 epsilon=0.025, 
                 theta=8, 
                 eta=1.0)

    # Load SAC agent
    agent = SAC(env.observation_space.shape, env.action_space, args)
    agent.load_checkpoint(args.checkpoint_path, evaluate=True)

    # Evaluate the model
    evaluate_model(env, agent, num_episodes=args.num_episodes, save_dir=args.save_dir)

    print(f"Evaluation completed. Images are saved in {args.save_dir}")

