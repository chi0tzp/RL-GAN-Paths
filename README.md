# RL-GAN-PATHS

## Overview
This project aims to explore meaningful paths in the pretrained StyleGAN2 latent space using reinforcement learning. By leveraging the Soft Actor-Critic (SAC) algorithm, the agent learns to navigate the latent space to generate images that align with a given text prompt while maintaining the identity of the initial image.

## Project Structure

- env_gan.py: Defines the custom environment interfacing with the StyleGAN2 model and handles the reward computation using CLIP and identity loss.
- sac.py: Implements the Soft Actor-Critic algorithm for continuous action spaces.
- model.py: Contains the neural network architectures for the policy and Q-value networks.
- replay_buffer.py: Implements different types of replay buffers, including Prioritized Experience Replay (PER).
- main.py: The main script to train the agent.
- utils.py: Utility functions used across the project.
- requirements.txt: Lists all Python dependencies.
- download_models.py: Script to download pretrained models required for the project.

## Installation

We recommend installing the required packages using python's native virtual environment as follows:

```bash
# Create a virtual environment and activate it
virtualenv --python 3.10 wgs-rl-venv
source wgs-rl-venv/bin/activate

# Upgrade pip and install requirements
pip install --upgrade pip
pip install setuptools
pip install -r requirements.txt

# Install CLIP (for obtaining the CLIP and FaRL ViT features)
pip install git+https://github.com/openai/CLIP.git
```

For using the aforementioned virtual environment in a Jupyter Notebook, you need to manually add the kernel as follows:

```bash
python -m ipykernel install --user --name=wgs-rl-venv
```



## Download pretrained models

```bash
python download_models.py
```

## Usage
To train the agent, run the main.py script with your desired configuration. Below are examples and explanations of how to use the script.

```bash
python main.py --text_prompt 'A photo of a smiling person' --cuda
```
This command starts training the agent to generate images of "A smiling person" using GPU acceleration.

### Command-Line Arguments
You can customize the training process using various command-line arguments:

- --policy: Type of policy network to use (Gaussian or Deterministic). Default: Gaussian.
- --memory: Type of replay memory (Tensor or PER). Default: Tensor.
- --gamma: Discount factor for rewards. Default: 0.99.
- --tau: Target smoothing coefficient for soft updates. Default: 0.005.
- --lr: Learning rate for the optimizer. Default: 0.0003.
- --alpha: Temperature parameter for entropy in SAC. Default: 0.2.
- --automatic_entropy_tuning: Enable automatic adjustment of alpha. Default: False.
- --seed: Random seed for reproducibility. Default: 123456.
- --batch_size: Batch size for training. Default: 256.
- --num_steps: Maximum number of training steps. Default: 1000001.
- --hidden_size: Hidden layer size in neural networks. Default: 256.
- --start_steps: Number of initial steps with random actions. Default: 5000.
- --target_update_interval: Frequency of target network updates. Default: 1.
- --replay_size: Capacity of the replay buffer. Default: 1000000.
- --cuda: Use CUDA for computation if available.
- --truncation: Maximum steps per episode before truncation. Default: 50.
- --gan_model_name: Name of the StyleGAN2 model to use. Default: 'stylegan2_ffhq512'.
- --text_prompt: Text prompt to guide image generation.
- --threshold: Similarity threshold for episode termination. Default: 1.6.
- --epsilon: Step size for action scaling. Default: 0.025.
- --theta: Number of initial layers in the GAN to modify. Default: 8.
- --eta: Weight for the identity loss in the reward function. Default: 1.
- --checkpoint_interval: How often to save model checkpoints (in steps). Default: 1000.
- --resume: Resume training from the last checkpoint.
- --wandb_log: Enable logging to Weights & Biases (W&B).
- --log_img_interval: Interval for logging generated images (in episodes). Default: 1000.

### Monitoring Training
- Weights & Biases (W&B): If enabled with --wandb_log, you can monitor loss curves, rewards, and generated images in real-time.
- Console Output: The script prints out training progress, including rewards and losses.
- Checkpoints: Model checkpoints are saved in the checkpoints/ directory at intervals specified by --checkpoint_interval.

Generated images are logged at intervals specified by --log_img_interval. If using W&B, they will appear under the "Media" section of your run. Otherwise, you can modify the code to save images locally.