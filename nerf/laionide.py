from PIL import Image
import torch as th
import torch.nn as nn

from glide_text2im.clip.model_creation import create_clip_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)
from glide_text2im.tokenizer.simple_tokenizer import SimpleTokenizer

import torch
import torch.nn.functional as F
import time
import os

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class Laionide(nn.Module):
    def __init__(self, device, ld_path="./laionide-v3-base.pt"):
        super().__init__()

        self.device = device

        print('[INFO] loading laionide...')
        
        # Create base model.

        options = model_and_diffusion_defaults()
        options['use_fp16'] = "cuda" in str(device) and False
        self.min_step = 20
        self.max_step = 980
        options['timestep_respacing'] = "1000"  # use 1000 diffusion steps for slow sampling
        self.model, self.diffusion = create_model_and_diffusion(**options)

        if len(ld_path) > 0:
            assert os.path.exists(
                ld_path
            ), f"Failed to resume from {ld_path}, file does not exist."
            weights = th.load(ld_path, map_location="cpu")
            self.model, self.diffusion = create_model_and_diffusion(**options)
            self.model.load_state_dict(weights)
            print(f"Resumed from {ld_path} successfully.")
        else:
            self.model, self.diffusion = create_model_and_diffusion(**options)
            self.model.load_state_dict(load_checkpoint("base", device))
        self.model.eval()
        if options["use_fp16"]:
            self.model.convert_to_fp16()
        self.model.to(device)
        print('total base parameters', sum(x.numel() for x in self.model.parameters()))
        self.options = options
        print(f'[INFO] loaded Laionide!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        tokens = list(map(self.model.tokenizer.encode, prompt))
        tokens, mask = map(list, zip(*[self.model.tokenizer.padded_tokens_and_mask(
            t,
            self.options["text_ctx"]
        ) for t in tokens]))
        neg_tokens = list(map(self.model.tokenizer.encode, negative_prompt))
        neg_tokens, neg_mask = map(list, zip(*[self.model.tokenizer.padded_tokens_and_mask(
            [],  # neg_tokens,
            self.options["text_ctx"]
        ) for nt in neg_tokens]))
        model_kwargs = dict(
            tokens=th.tensor(
                neg_tokens + tokens, device=self.device
            ),
            mask=th.tensor(
                neg_mask + mask,
                dtype=th.bool,
                device=self.device,
            ),
        )
        return model_kwargs


    def train_step(self, text_embeddings, pred_rgb, guidance_scale=100):
        
        # interp to 64x64 to be fed into diffusion.

        # _t = time.time()
        latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False) * 2 - 1
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.diffusion.q_sample(latents, t, noise=noise).to(self.device)
            # pred noise
            latent_model_input = latents_noisy.repeat(2, 1, 1, 1)
            model_out = self.model(latent_model_input, t, **text_embeddings)
            noise_pred, _ = model_out[:, :3], model_out[:, 3:]
            # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

            # perform guidance (high scale from paper!)
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        alpha = self.diffusion.alphas_cumprod[t]  # self.alphas[t]
        w = (1 - alpha)
        # w = alpha ** 0.5 * (1 - alpha)
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0 # dummy loss value

    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        # posterior = self.vae.encode(imgs).latent_dist
        latents = imgs  # posterior.sample() * 0.18215

        return latents

    """
    def prompt_to_img(self, prompts, negative_prompts='', height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):

        if isinstance(prompts, str):
            prompts = [prompts]
        
        if isinstance(negative_prompts, str):
            negative_prompts = [negative_prompts]

        # Prompts -> text embeds
        text_embeds = self.get_text_embeds(prompts, negative_prompts) # [2, 77, 768]

        # Text embeds -> img latents
        latents = self.produce_latents(text_embeds, height=height, width=width, latents=latents, num_inference_steps=num_inference_steps, guidance_scale=guidance_scale) # [1, 4, 64, 64]
        
        # Img latents -> imgs
        imgs = self.decode_latents(latents) # [1, 3, 512, 512]

        # Img to Numpy
        imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
        imgs = (imgs * 255).round().astype('uint8')

        return imgs


if __name__ == '__main__':

    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('prompt', type=str)
    parser.add_argument('--negative', default='', type=str)
    parser.add_argument('--sd_version', type=str, default='2.0', choices=['1.5', '2.0'], help="stable diffusion version")
    parser.add_argument('-H', type=int, default=512)
    parser.add_argument('-W', type=int, default=512)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--steps', type=int, default=50)
    opt = parser.parse_args()

    seed_everything(opt.seed)

    device = torch.device('cuda')

    sd = StableDiffusion(device, opt.sd_version)

    imgs = sd.prompt_to_img(opt.prompt, opt.negative, opt.H, opt.W, opt.steps)

    # visualize image
    plt.imshow(imgs[0])
    plt.show()
    """

