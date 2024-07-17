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
from lib.id import IDLoss

class GANEnv(gym.Env):

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, gan_model_name, text_prompt, eval_mode=False, epsilon=0.1, threshold=0.6, batch_size=1, eta=1, theta=8, truncation = 0.7, use_cuda=True, render_mode=None):
        super(GANEnv, self).__init__()

        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = "cuda" if self.use_cuda else "cpu"

        self.eval_mode = eval_mode

        self.truncation = truncation # W-space truncation parameter

        self.epsilon = epsilon # Module of the step
        self.threshold = threshold # Threshold of similarity for termination
        #self.eta = eta # Weight of reward_id
        self.theta = theta # First layers of GAN that will be modified by actions -> # first layers are the one responsible for general modification while the last layers could impact color palette and some details we do not want to change
        #self.batch_size = batch_size  # Number of images to process in a batch

        self.episode = 1

        # ID
        #self.id_loss = IDLoss() # this need cuda
        #self.id_transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256)])

        # GAN
        self.gan_model_name = gan_model_name
        self.G = load_generator(model_name=self.gan_model_name, latent_is_w=True).eval()

        if self.use_cuda:
            self.G = self.G.cuda()

        # CLIP
        self.clip_model, _ = clip.load("ViT-B/16", device=self.device)
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
        self.text_features = self.clip_model.encode_text(self.text_inputs)

        # Assert we can modify #self.theta layers
        assert self.G.num_layers >= self.theta
        self.default_layers = self.G.num_layers - self.theta

        # Action and observation space
        self.latent_dim = self.G.num_layers * self.G.dim_z # num_layers X img_size (dim_z)
        self.action_space = spaces.Box(low=-100000, high=100000, shape=(self.theta, self.G.dim_z), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.G.num_layers, self.G.dim_z), dtype=np.float32)

        z = torch.randn(1, self.G.dim_z).to(self.device)
        wp = self.G.get_w(z, truncation=self.truncation)
        self.current_latent = wp
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

    def step(self, action):
        delta = torch.tensor(action, dtype=torch.float32)

        if len(delta.shape) == 2:
            delta = delta.unsqueeze(0)

        if self.use_cuda:
            delta = delta.cuda()

        # L2 normalisation
        delta_normalised = F.normalize(delta, p=2, dim=1)

        # Check if delta is L2 normalised
        l2_norms = torch.norm(delta_normalised, p=2, dim=1)
        assert torch.allclose(l2_norms, torch.ones_like(l2_norms), atol=1e-6) == True

        step_vector = self.epsilon * delta_normalised # shape: [batch, theta, img_size]
        
        # Expand the step_vector to match current_latent
        zeros = torch.zeros(step_vector.shape[0], self.default_layers, step_vector.shape[-1]).to(self.device)
        step_vector = torch.cat((step_vector, zeros), dim=1)
        
        self.current_latent = self.current_latent + step_vector # shape: [batch, num_layers, img_size]

        generated_image = self.G(self.current_latent)
        #image = tensor2image(generated_image.cpu(), adaptive=True)

        # Reward CLIP
        image_features = self.clip_model.encode_image(self.clip_img_transform(generated_image))
        reward_clip = torch.cosine_similarity(image_features, self.text_features.unsqueeze(0)).mean().item()

        # Reward ID
        #reward_id = self.id_loss(y_hat=self.id_transform(generated_image), y=self.id_transform(self.previous_image))
        
        reward = reward_clip #+ self.eta * reward_id

        # Observation
        observation = self.current_latent.clone().detach().cpu().numpy()

        terminated = reward >= self.threshold
        truncated = self.episode >= 50 # Truncation after 50 steps
        if self.eval_mode:
            image = tensor2image(generated_image.cpu(), adaptive=True)
            info = {"reward": reward, "image": image, "step": self.episode}
        else:
            info = {}

        self.episode += 1

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset()        

        self.episode = 0

        z = torch.randn(1, self.G.dim_z).to(self.device)
        wp = self.G.get_w(z, truncation=self.truncation)
        self.current_latent = wp
        if self.use_cuda:
            self.current_latent = self.current_latent.cuda()

        observation = self.current_latent.clone().detach().cpu().numpy()

        return observation, {}

if __name__ == "__main__":
    env = GANEnv(gan_model_name='stylegan2_ffhq512', text_prompt='test prompt', threshold=0.95, epsilon=0.05)

    done = False
    observation, _ = env.reset(seed=42)
    action = env.action_space.sample()
    action = torch.tensor(action, dtype=torch.float32)

    while not done:
        observation, reward, terminated, truncated, info = env.step(action)
        done = truncated or terminated
        print(f"Reward: {reward}")
        images = info["image"]
        plt.imshow(images[0])
        plt.show()