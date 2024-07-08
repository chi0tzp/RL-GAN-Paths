import gymnasium as gym
from hashlib import sha1
import numpy as np
import torch
from gymnasium import spaces
from lib.utils import tensor2image
import matplotlib.pyplot as plt
import clip
from models.load_generator import load_generator
import torch.nn.functional as F
from torchvision import transforms
import os

class GANEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gan_model_name, text_prompt, epsilon=0.1, threshold=0.6, truncation = 0.7, use_cuda=True, render_mode=None):
        super(GANEnv, self).__init__()

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        self.truncation = truncation # W-space truncation parameter

        self.epsilon = epsilon # Module of the step
        self.threshold = threshold # Threshold of similarity for termination

        self.episode = 1

        # GAN
        self.gan_model_name = gan_model_name
        self.G = load_generator(model_name=self.gan_model_name, latent_is_w=True).eval()

        if self.use_cuda:
            self.G = self.G.cuda()

        # CLIP
        self.clip_model, _ = clip.load("ViT-B/16", device='cpu')
        farl_model = os.path.join('models/pretrained/farl/FaRL-Base-Patch16-LAIONFace20M-ep64.pth')
        farl_state = torch.load(farl_model)
        self.clip_model.load_state_dict(farl_state["state_dict"], strict=False)
        self.clip_model.float()
        self.clip_model.eval()

        # CLIP transforms
        self.clip_img_transform = transforms.Compose([transforms.Resize(224),
                                                 transforms.CenterCrop(224),
                                                 transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                                      (0.26862954, 0.26130258, 0.27577711))])

        self.text_inputs = clip.tokenize([text_prompt]).to(self.device)

        # Action and observation space
        self.latent_dim = self.G.num_layers * self.G.dim_z # num_layers X img_size (dim_z)
        self.action_space = spaces.Box(low=-1000, high=1000, shape=(self.latent_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)

        z = torch.randn(1, self.G.dim_z)
        wp = self.G.get_w(z, truncation=self.truncation)
        self.current_latent = wp.flatten(start_dim=1, end_dim=-1)
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

    def step(self, action):

        delta = torch.tensor(action, dtype=torch.float32)
        if self.use_cuda:
            delta = delta.cuda()

        # L2 normalisation
        delta_normalised = F.normalize(delta, p=2, dim=0)

        step_vector = self.epsilon * delta_normalised
        self.current_latent = self.current_latent + step_vector

        generated_image = self.G(self.current_latent)
        image = tensor2image(generated_image.cpu(), adaptive=True)

        # Reward
        image_features = self.clip_model.encode_image(self.clip_img_transform(generated_image))
        text_features = self.clip_model.encode_text(self.text_inputs)
        reward = torch.cosine_similarity(image_features, text_features).item()

        # Observation
        observation = self.current_latent.detach().cpu().numpy()

        terminated = reward >= self.threshold
        truncated = self.episode >= 50 # Truncation after 50 steps
        info = {"reward": reward, "image": image, "step": self.episode}

        self.episode += 1

        return observation, reward, terminated, truncated, info
    
    def compute_reward(self, achived_goal, required_goal, info):
        return 1

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.episode = 0

        z = torch.randn(1, self.G.dim_z)
        wp = self.G.get_w(z, truncation=self.truncation)
        self.current_latent = wp.flatten(start_dim=1, end_dim=-1)
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

        observation = self.current_latent.detach().cpu().numpy()

        return observation, {}

if __name__ == "__main__":
    env = GANEnv(gan_model_name='stylegan2_ffhq512', text_prompt='animated', threshold=0.95, epsilon=0.05)

    done = False
    observation, _ = env.reset(seed=42)
    action = env.action_space.sample()

    while not done:
        observation, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        print(f"Reward: {reward}")
        image = info["image"]
        plt.imshow(image)
        plt.show()