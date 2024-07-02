import gymnasium as gym
from hashlib import sha1
import numpy as np
import torch
from gymnasium import spaces
from lib.utils import tensor2image
import matplotlib.pyplot as plt
import clip
from models.load_generator import load_generator

class GANEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gan_model_name, text_prompt, epsilon=0.1, threshold=0.6, truncation = 0.7, use_cuda=True, render_mode=None):
        super(GANEnv, self).__init__()

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        self.truncation = truncation # W-space truncation parameter

        self.epsilon = epsilon # Module of the step
        self.threshold = threshold # Threshold of similarity for termination

        # GAN
        self.gan_model_name = gan_model_name
        self.G = load_generator(model_name=self.gan_model_name, latent_is_s='stylegan' in gan_model_name).eval()

        if self.use_cuda:
            self.G = self.G.cuda()

        # CLIP
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)
        self.text_inputs = clip.tokenize([text_prompt]).to(self.device)

        # Action and observation space
        self.latent_dim = self.G.dim_z
        self.action_space = spaces.Box(low=-np.pi, high=np.pi, shape=(self.latent_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.latent_dim,), dtype=np.float32)

        self.current_latent = torch.randn(1, self.latent_dim)
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

    def generate_image(self, latent_code):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z = latent_code
        latent_code_hash = sha1(z.cpu().numpy()).hexdigest()

        if 'stylegan' in self.gan_model_name:
            # Get W/W+ latent codes from z code
            wp = self.G.get_w(z, truncation=self.truncation)
            # Get S latent codes from wp codes
            styles_dict = self.G.get_s(wp)

            # Generate image
            with torch.no_grad():
                img = self.G(styles_dict)
        else:
            # Generate image
            with torch.no_grad():
                img = self.G(z)

        return img

    def step(self, action):

        delta = torch.tensor(action, dtype=torch.float32)
        if self.use_cuda:
            delta = delta.cuda()

        step_vector = self.epsilon * torch.sin(delta)
        
        self.current_latent = self.current_latent + step_vector

        with torch.no_grad():
            generated_image = self.generate_image(self.current_latent)

        image = tensor2image(generated_image.cpu(), adaptive=True)
        image_input = self.clip_preprocess(image).unsqueeze(0).to(self.device)

        # Reward
        with torch.no_grad():
            image_features = self.clip_model.encode_image(image_input)
            text_features = self.clip_model.encode_text(self.text_inputs)
            reward = torch.cosine_similarity(image_features, text_features).item()

        # Observation
        observation = self.current_latent.cpu().numpy()

        terminated = reward >= self.threshold
        truncated = False
        info = {"reward": reward, "image": image}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_latent = torch.randn(1, self.latent_dim)
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

        observation = self.current_latent.cpu().numpy()

        return observation, {}

    def render(self):
        pass

    def close(self):
        pass


if __name__ == "__main__":
    env = GANEnv(gan_model_name='stylegan2_ffhq512', text_prompt='animated', threshold=0.95, epsilon=0.25)

    done = False
    observation, _ = env.reset(seed=42)
    action = env.action_space.sample()

    while not done:
        observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
        done = truncated or terminated
        print(f"Reward: {reward}")
        image = info["image"]
        plt.imshow(image)
        plt.show()