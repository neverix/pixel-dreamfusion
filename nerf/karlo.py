from transformers import logging
from diffusers import UnCLIPPipeline

# suppress partial model loading warning
logging.set_verbosity_error()

import torch
import torch.nn as nn
import torch.nn.functional as F

import time

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = True

class Karlo(nn.Module):
    def __init__(self, device):
        super().__init__()

        self.device = device

        print(f'[INFO] loading karlo...')

        # Create model
        self.pipe = UnCLIPPipeline.from_pretrained("kakaobrain/karlo-v1-alpha").to(device)
        self.text_proj = self.pipe.text_proj
        self.prior_scheduler = self.pipe.prior_scheduler
        self.prior = self.pipe.prior
        self.unet = self.pipe.decoder
        self.scheduler = self.pipe.decoder_scheduler
        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * 0.98)
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience
        print(f'[INFO] loaded karlo!')

    def get_text_embeds(self, prompt, negative_prompt):
        generator = None
        text_embeddings, text_encoder_hidden_states, text_mask = self.pipe._encode_prompt(
            prompt, self.device, 1, True
        )
        
        # prior
        self.prior_scheduler.set_timesteps(50, device=self.device)
        prior_timesteps_tensor = self.prior_scheduler.timesteps

        embedding_dim = self.prior.config.embedding_dim
        prior_latents = self.prepare_latents(
            (1, embedding_dim),
            text_embeddings.dtype,
            self.device,
            generator,
            prior_latents,
            self.prior_scheduler,
        )

        for i, t in enumerate(self.pipe.progress_bar(prior_timesteps_tensor)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([prior_latents] * 2)

            predicted_image_embedding = self.prior(
                latent_model_input,
                timestep=t,
                proj_embedding=text_embeddings,
                encoder_hidden_states=text_encoder_hidden_states,
                attention_mask=text_mask,
            ).predicted_image_embedding

            predicted_image_embedding_uncond, predicted_image_embedding_text = predicted_image_embedding.chunk(2)
            predicted_image_embedding = predicted_image_embedding_uncond + prior_guidance_scale * (
                predicted_image_embedding_text - predicted_image_embedding_uncond
            )

            if i + 1 == prior_timesteps_tensor.shape[0]:
                prev_timestep = None
            else:
                prev_timestep = prior_timesteps_tensor[i + 1]

            prior_latents = self.prior_scheduler.step(
                predicted_image_embedding,
                timestep=t,
                sample=prior_latents,
                generator=generator,
                prev_timestep=prev_timestep,
            ).prev_sample

        prior_latents = self.prior.post_process_latents(prior_latents)

        image_embeddings = prior_latents
        return text_embeddings, text_encoder_hidden_states, text_mask, image_embeddings

    def train_step(self, embeddings, pred_rgb, guidance_scale=100):
        text_embeddings, text_encoder_hidden_states, text_mask, image_embeddings = embeddings
        decoder_text_mask = F.pad(text_mask, (self.text_proj.clip_extra_context_tokens, 0), value=1)
        text_encoder_hidden_states, additive_clip_time_embeddings = self.text_proj(
            image_embeddings=image_embeddings,
            text_embeddings=text_embeddings,
            text_encoder_hidden_states=text_encoder_hidden_states,
            do_classifier_free_guidance=True,
        )
        
        # interp to 512x512 to be fed into vae.

        # _t = time.time()
        latents = F.interpolate(pred_rgb, (64, 64), mode='bilinear', align_corners=False)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: interp {time.time() - _t:.4f}s')

        # timestep ~ U(0.02, 0.98) to avoid very high/low noise level
        t = torch.randint(self.min_step, self.max_step + 1, [1], dtype=torch.long, device=self.device)

        # encode image into latents with vae, requires grad!
        # _t = time.time()
        # latents = self.encode_imgs(pred_rgb_64)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()
        with torch.no_grad():
            # add noise
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)
            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            noise_pred = self.unet(
                sample=latent_model_input,
                timestep=t,
                encoder_hidden_states=text_encoder_hidden_states,
                class_labels=additive_clip_time_embeddings,
                attention_mask=decoder_text_mask).sample
        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t])
        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])
        grad = w * (noise_pred - noise)

        # clip grad for stable training?
        # grad = grad.clamp(-10, 10)
        grad = torch.nan_to_num(grad)

        # manually backward, since we omitted an item in grad and cannot simply autodiff.
        # _t = time.time()
        latents.backward(gradient=grad, retain_graph=True)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: backward {time.time() - _t:.4f}s')

        return 0  # dummy loss value



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



