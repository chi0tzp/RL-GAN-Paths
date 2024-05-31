# WarpedGANSpace-RL-dev

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
(wgs-rl-venv) $ python -m ipykernel install --user --name=wgs-rl-venv
```



## Download pretrained models

```bash
python download_models.py
```







