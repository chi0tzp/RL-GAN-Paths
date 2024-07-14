from PIL import Image
from stable_baselines3.common.callbacks import BaseCallback
import os

class CustomCallback(BaseCallback):
    def __init__(self, save_freq, save_path, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            image = self.locals['infos'][0]["image"]
            step = self.num_timesteps
            image_path = os.path.join(self.save_path, f'step={step}.jpg')
            image.save(image_path)
        return True