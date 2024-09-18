import argparse
import datetime
import os
import numpy as np
import itertools
import torch
import wandb
from sac import SAC
from replay_memory import ReplayMemory, TensorReplayMemory, PrioritizedReplayMemory
from env_gan import GANEnv

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')

# SAC args
parser.add_argument('--policy', default="Gaussian", help='Policy Type: Gaussian | Transformer | Deterministic (default: Gaussian)')
parser.add_argument('--memory', default="Tensor", help='Memory Type: Tensor | PER (default: Tensor)')
parser.add_argument('--eval', action='store_true', help='Evaluates a policy every 10 episodes (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G', help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G', help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G', help='Temperature parameter α determines the relative importance of the entropy term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', action='store_true', help='Automatically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N', help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N', help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N', help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N', help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=5000, metavar='N', help='Steps sampling random actions (default: 5000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N', help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N', help='size of replay buffer (default: 1000000)')
parser.add_argument('--cuda', action="store_true", help='run on CUDA (default: False)')
parser.add_argument('--truncation', type = int, help = "Number of episodes for the truncation of the trajectory (default: 50)", default=50)

# GANEnv args
parser.add_argument('--gan_model_name', type = str, help = 'Name of the model.', default='stylegan2_ffhq512')
parser.add_argument('--text_prompt', type = str, help = 'Text prompt.', default='Test prompt')
parser.add_argument('--threshold', type = float, help = 'Threshold of similarity for termination.', default=1.6)
parser.add_argument('--epsilon', type = float, help = 'Module of the step.', default=0.025)
parser.add_argument('--theta', type = int, help = 'Number of the first layers of GAN that will be modified by actions', default=8)
parser.add_argument('--eta', type = float, help = 'Weight of IDloss reward (reward_id).', default=1)
parser.add_argument('--checkpoint_interval', type=int, default=1000, metavar='N', help='How often to save checkpoints (default: 10000 iterations)')
parser.add_argument('--resume', action="store_true", help='Resume training from the last checkpoint (default: False)')
parser.add_argument('--wandb_log', action="store_true", help='Log in Wandb (default: False)')
parser.add_argument('--log_img_interval', type=int, default=1000, metavar='N', help='How often to save generated images (default: 1000 episodes)')

args = parser.parse_args()

project_name = f"{args.text_prompt}-policy={args.policy}-mem={args.memory}-eta={args.eta}-threshold={args.threshold}-ep={args.epsilon}-batch={args.batch_size}-lr={args.lr}-gamma={args.gamma}-tau={args.tau}-alpha={args.alpha}"

# Initialize Weights & Biases
if args.wandb_log: wandb.init(project="RL-GAN", name=project_name, config=args)

# ENV
env = GANEnv(gan_model_name=args.gan_model_name, 
                 text_prompt=args.text_prompt,
                 threshold=args.threshold, 
                 epsilon=args.epsilon,
                 theta=args.theta, 
                 eta=args.eta)

# Set environment seed
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Initialize SAC agent
agent = SAC(env.observation_space.shape, env.action_space, args)

# Resume training if checkpoint exists and resume flag is set to True
if args.resume:
    checkpoint_path = "checkpoints/sac_checkpoint_{}_latest.pth".format(project_name)
    if os.path.isfile(checkpoint_path):
        agent.load_checkpoint(checkpoint_path)
        print("Resumed training from checkpoint: {}".format(checkpoint_path))
    else:
        print("Checkpoint not found. Starting from scratch.")

# Memory
#memory = ReplayMemory(args.replay_size, args.seed)
if args.memory == 'Tensor':
    memory = TensorReplayMemory(args.replay_size, state_shape=env.observation_space.shape, action_shape=env.action_space.shape, device='cuda' if args.cuda else 'cpu')
elif args.memory == 'PER':
    memory = PrioritizedReplayMemory(args.replay_size, state_shape=env.observation_space.shape, action_shape=env.action_space.shape, device='cuda' if args.cuda else 'cpu')

# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    episode_reward_clip = 0
    episode_reward_id = 0
    done = False
    state = env.reset()
    log_img = False
    if i_episode % args.log_img_interval == 0: log_img = True
    beta_end = 1.0
    beta_start = 0.4

    beta = beta_start

    while not done:
        if total_numsteps < args.start_steps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        # Perform update step
        if len(memory) > args.batch_size: # If we collected enough transitions
            for _ in range(args.updates_per_step):
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates, beta=beta)
                #eta_t = eta_0 + (eta_T - eta_0) * (total_numsteps / args.num_steps)
                #critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory, args.batch_size, updates, eta=eta_t, K=args.updates_per_step)

                # Update beta
                beta = min(beta_end, beta_start + (1.0 - beta_start) * updates / args.num_steps)

                if args.wandb_log:
                    wandb.log({
                        'loss/critic_1': critic_1_loss,
                        'loss/critic_2': critic_2_loss,
                        'loss/policy': policy_loss,
                        'loss/entropy_loss': ent_loss,
                        'entropy_temperature/alpha': alpha,
                        'beta': beta
                        })
                updates += 1

                # Save checkpoint at specified intervals
                if updates % args.checkpoint_interval == 0:
                    agent.save_checkpoint(project_name, suffix="latest")

        # Environment step
        next_state, reward, reward_clip, reward_id, terminated, info = env.step(action, log_img)
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward
        episode_reward_clip += reward_clip
        episode_reward_id += reward_id 
        truncated = episode_steps >= args.truncation
        done = truncated or terminated
        # Mask for terminal state
        mask = 1 if episode_steps == env.max_episode_steps else float(not done)

        memory.push(state, action, reward, next_state, mask)

        state = next_state

        if log_img and args.wandb_log: wandb.log({'image': wandb.Image(info['image'], caption=f"CLIP:{reward_clip:.2f}-ID:{reward_id:.2f}")})
        if args.wandb_log: wandb.log({'reward_per_episode/total': reward, 'reward_per_episode/clip': reward_clip, 'reward_per_episode/id': reward_id})

    if args.wandb_log: wandb.log({'reward/train': episode_reward, 'reward/reward_clip': episode_reward_clip, 'reward/reward_id': episode_reward_id})

    if total_numsteps > args.num_steps:
        break

    # Evaluation
    if i_episode % 10 == 0 and args.eval:
        avg_reward = 0.0
        episodes = 10
        for _ in range(episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = agent.select_action(state, evaluate=True)
                next_state, reward, reward_clip, reward_id, terminated, info = env.step(action, log_img=True)
                episode_reward += reward
                state = next_state
            avg_reward += episode_reward
        avg_reward /= episodes
        if args.wandb_log: 
            wandb.log({'avg_reward/test': avg_reward}, step=total_numsteps)
            wandb.log({'image': wandb.Image(info['image'])})
        print("----------------------------------------")
        print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
        print("----------------------------------------")

env.close()
if args.wandb_log: wandb.finish()
