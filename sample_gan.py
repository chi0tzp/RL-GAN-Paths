import os
import os.path as osp
import argparse
import torch
import json
from hashlib import sha1
from lib import GENFORCE_MODELS, update_progress, update_stdout, tensor2image
from models.load_generator import load_generator


def main():
    """A script for sampling from a pre-trained GAN's latent space and generating images. The generated images, along
    with the corresponding latent codes, will be stored under `experiments/latent_codes/<gan>/`.

    Options:
        -v, --verbose : set verbose mode on
        --gan         : set GAN generator (see GENFORCE_MODELS in lib/config.py)
        --truncation  : set W-space truncation parameter. If set, W-space codes will be truncated
        --num-samples : set the number of latent codes to sample for generating images
        --cuda        : use CUDA (default)
        --no-cuda     : do not use CUDA
    """
    parser = argparse.ArgumentParser(description="Sample a pre-trained GAN latent space and generate images")
    parser.add_argument('-v', '--verbose', action='store_true', help="verbose mode on")
    parser.add_argument('--gan', type=str, required=True, choices=GENFORCE_MODELS.keys(), help='GAN generator')
    parser.add_argument('--truncation', type=float, default=0.7, help="W-space truncation parameter")
    parser.add_argument('--num-samples', type=int, default=4, help="set number of latent codes to sample")
    parser.add_argument('--cuda', dest='cuda', action='store_true', help="use CUDA during training")
    parser.add_argument('--no-cuda', dest='cuda', action='store_false', help="do NOT use CUDA during training")
    parser.set_defaults(cuda=True)

    # Parse given arguments
    args = parser.parse_args()

    # Create output dir for generated images
    out_dir = osp.join('experiments', 'latent_codes', args.gan)
    out_dir = osp.join(out_dir, '{}-{}'.format(args.gan, args.num_samples))
    os.makedirs(out_dir, exist_ok=True)

    # Save argument in json file
    with open(osp.join(out_dir, 'args.json'), 'w') as args_json_file:
        json.dump(args.__dict__, args_json_file)

    # CUDA
    use_cuda = False
    if torch.cuda.is_available():
        if args.cuda:
            use_cuda = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            print("*** WARNING ***: It looks like you have a CUDA device, but aren't using CUDA.\n"
                  "                 Run with --cuda for optimal training speed.")
            torch.set_default_tensor_type('torch.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')

    # Build GAN generator model and load with pre-trained weights
    if args.verbose:
        print("#. Build GAN generator model G and load with pre-trained weights...")
        print("  \\__GAN generator : {} (res: {})".format(args.gan, GENFORCE_MODELS[args.gan][1]))
        print("  \\__Pre-trained weights: {}".format(GENFORCE_MODELS[args.gan][0]))

    G = load_generator(model_name=args.gan, latent_is_s='stylegan' in args.gan, verbose=args.verbose).eval()

    # Upload GAN generator model to GPU
    if use_cuda:
        G = G.cuda()

    # Latent codes sampling
    if args.verbose:
        print("#. Sample {} {}-dimensional latent codes...".format(args.num_samples, G.dim_z))
    zs = torch.randn(args.num_samples, G.dim_z)

    if use_cuda:
        zs = zs.cuda()

    if args.verbose:
        print("#. Generate images...")
        print("  \\__{}".format(out_dir))

    # Iterate over given latent codes
    for i in range(args.num_samples):
        # Un-squeeze current latent code in shape [1, dim] and create hash code for it
        z = zs[i, :].unsqueeze(0)
        latent_code_hash = sha1(z.cpu().numpy()).hexdigest()

        if args.verbose:
            update_progress(
                "  \\__.Latent code hash: {} [{:03d}/{:03d}] ".format(latent_code_hash, i + 1, args.num_samples),
                args.num_samples, i)

        # Create directory for current latent code
        latent_code_dir = osp.join(out_dir, '{}'.format(latent_code_hash))
        os.makedirs(latent_code_dir, exist_ok=True)

        if 'stylegan' in args.gan:
            # Get W/W+ latent codes from z code
            wp = G.get_w(z, truncation=args.truncation)
            w = wp[:, 0, :]
            # Get S latent codes from wp codes
            styles_dict = G.get_s(wp)

            # Generate image
            with torch.no_grad():
                img_s = G(styles_dict)
            tensor2image(img_s.cpu(), adaptive=True).save(osp.join(latent_code_dir, 'image_s.jpg'),
                                                          "JPEG", quality=100, subsampling=0, progressive=True)
            # Save latent codes in Z/W/W+/S spaces
            torch.save(z.cpu(), osp.join(latent_code_dir, 'latent_code_z.pt'))
            torch.save(w.cpu(), osp.join(latent_code_dir, 'latent_code_w.pt'))
            torch.save(styles_dict, osp.join(latent_code_dir, 'latent_code_s.pt'))
        else:
            # Generate image
            with torch.no_grad():
                img_z = G(z)
            tensor2image(img_z.cpu(), adaptive=True).save(osp.join(latent_code_dir, 'image_z.jpg'),
                                                          "JPEG", quality=100, subsampling=0, progressive=True)
            # Save latent code in Z-space
            torch.save(z.cpu(), osp.join(latent_code_dir, 'latent_code_z.pt'))

    if args.verbose:
        update_stdout(1)
        print()
        print()


if __name__ == '__main__':
    main()
