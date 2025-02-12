# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
"""Project given image to the latent space of pretrained network pickle."""

import copy
import os
from tqdm import tqdm
from time import perf_counter

import click
import imageio
import numpy as np
import PIL.Image
import torch
import torch.nn.functional as F

import dnnlib
import legacy


def project(
        G,
        target: torch.
    Tensor,  # [C,H,W] and dynamic range [0,255], W & H must match G output resolution
        *,
        feature_extractor_model,
        num_steps=1000,
        w_avg_samples=10000,
        initial_learning_rate=0.1,
        initial_noise_factor=0.05,
        lr_rampdown_length=0.25,
        lr_rampup_length=0.05,
        noise_ramp_length=0.75,
        regularize_noise_weight=1e5,
        verbose=False,
        device: torch.device
    ):
    assert target.shape == (G.img_channels, G.img_resolution, G.img_resolution)

    def logprint(*args):
        if verbose:
            print(*args)

    G = copy.deepcopy(G).eval().requires_grad_(False).to(
        device
        )  # type: ignore

    # Compute w stats.
    logprint(
        f'Computing W midpoint and stddev using {w_avg_samples} samples...'
        )
    z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
    w_samples = G.mapping(
        torch.from_numpy(z_samples).to(device), None
        )  # [N, L, C]
    w_samples = w_samples[:, :1, :].cpu().numpy().astype(
        np.float32
        )  # [N, 1, C]
    w_avg = np.mean(w_samples, axis=0, keepdims=True)  # [1, 1, C]
    w_std = (np.sum((w_samples - w_avg)**2) / w_avg_samples)**0.5

    # Setup noise inputs.
    noise_bufs = {
        name: buf
        for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name
        }

    # Features for target image.
    target_images = target.unsqueeze(0).to(device).to(torch.float32)
    if target_images.shape[2] > 256:
        target_images = F.interpolate(
            target_images, size=(256, 256), mode='area'
            )
    target_features = feature_extractor_model(
        target_images, resize_images=False, return_lpips=True
        )

    w_opt = torch.tensor(
        w_avg, dtype=torch.float32, device=device, requires_grad=True
        )  # pylint: disable=not-callable
    w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]),
                        dtype=torch.float32,
                        device=device)
    optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()),
                                 betas=(0.9, 0.999),
                                 lr=initial_learning_rate)

    # Init noise.
    for buf in noise_bufs.values():
        buf[:] = torch.randn_like(buf)
        buf.requires_grad = True

    for step in range(num_steps):
        # Learning rate schedule.
        t = step / num_steps
        w_noise_scale = w_std * initial_noise_factor * max(
            0.0, 1.0 - t / noise_ramp_length
            )**2
        lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
        lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
        lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
        lr = initial_learning_rate * lr_ramp
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Synth images from opt_w.
        w_noise = torch.randn_like(w_opt) * w_noise_scale
        ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
        synth_images = G.synthesis(ws, noise_mode='const')

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        synth_images = (synth_images + 1) * (255 / 2)
        if synth_images.shape[2] > 256:
            synth_images = F.interpolate(
                synth_images, size=(256, 256), mode='area'
                )

        # Features for synth images.
        synth_features = feature_extractor_model(
            synth_images, resize_images=False, return_lpips=True
            )
        dist = (target_features - synth_features).square().sum()

        # Noise regularization.
        reg_loss = 0.0
        for v in noise_bufs.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise
                             * torch.roll(noise, shifts=1, dims=3)).mean()**2
                reg_loss += (noise
                             * torch.roll(noise, shifts=1, dims=2)).mean()**2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)
        loss = dist + reg_loss * regularize_noise_weight

        # Step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        logprint(
            f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}'
            )

        # Save projected W for each optimization step.
        w_out[step] = w_opt.detach()[0]

        # Normalize noise.
        with torch.no_grad():
            for buf in noise_bufs.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()

    return w_out.repeat([1, G.mapping.num_ws, 1])


#----------------------------------------------------------------------------


# yapf: disable
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--feature-ext', 'feature_extractor_pkl', help='Feature extractor model pickle filename', required=True)
@click.option('--target', 'target_fname', help='Target image file to project to', required=True, metavar='FILE')
@click.option('--name', 'output_name',    help='Name of the ouput file')
@click.option('--num-steps',              help='Number of optimization steps', type=int, default=1000, show_default=True)
@click.option('--seed',                   help='Random seed', type=int, default=303, show_default=True)
@click.option('--save-video',             help='Save an mp4 video of optimization progress', default=False, is_flag=True)
@click.option('--outdir',                 help='Where to save the output images', required=True, metavar='DIR')
@click.option('--fps',                    help='Frames per second of final video', default=30, show_default=True)
# yapf: enable
def run_projection(
    network_pkl: str,
    feature_extractor_pkl: str,
    target_fname: str,
    output_name: str,
    outdir: str,
    save_video: bool,
    seed: int,
    num_steps: int,
    fps: int,
    ):
    """Project given image to the latent space of pretrained network pickle.


    Examples:
    \b
    python projector.py --outdir=out --target=~/mytargetimg.png \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load networks.
    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as fp:
        G = legacy.load_network_pkl(fp)['G_ema'].requires_grad_(False).to(
            device
            )  # type: ignore

    # Load target image.
    target_pil = PIL.Image.open(target_fname).convert('RGB')
    w, h = target_pil.size
    s = min(w, h)
    target_pil = target_pil.crop(
        ((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)
        )
    target_pil = target_pil.resize((G.img_resolution, G.img_resolution),
                                   PIL.Image.LANCZOS)
    target_uint8 = np.array(target_pil, dtype=np.uint8)

    # Load VGG16 feature detector.
    print(f'Loading feature detection model from "{feature_extractor_pkl}"...')
    with dnnlib.util.open_url(feature_extractor_pkl) as f:
        vgg16 = torch.jit.load(f).eval().to(device)

    # Optimize projection.
    start_time = perf_counter()
    projected_w_steps = project(
        G,
        target=torch.tensor(target_uint8.transpose([2, 0, 1]), device=device),  # pylint: disable=not-callable
        feature_extractor_model=vgg16,
        num_steps=num_steps,
        device=device,
        verbose=True
        )
    print(f'Elapsed: {(perf_counter()-start_time):.1f} s')

    filename = output_name if output_name else "proj"
    target_filename = f'{output_name}_target' if output_name else '_target'
    vector_filename = f'{output_name}_projected_w' if output_name else 'projected_w'

    # Save final projected frame and W vector.
    os.makedirs(outdir, exist_ok=True)

    target_pil.save(os.path.join(outdir, f'{target_filename}.png'))
    projected_w = projected_w_steps[-1]
    synth_image = G.synthesis(projected_w.unsqueeze(0), noise_mode='const')
    synth_image = (synth_image + 1) * (255 / 2)
    synth_image = synth_image.permute(0, 2, 3,
                                      1).clamp(0, 255).to(torch.uint8
                                                          )[0].cpu().numpy()
    PIL.Image.fromarray(synth_image,
                        'RGB').save(os.path.join(outdir, f'{filename}.png'))
    np.savez(
        os.path.join(outdir, f'{vector_filename}.npz'),
        w=projected_w.unsqueeze(0).cpu().numpy()
        )

    # Render debug output: optional video and projected image and W vector.
    if save_video:
        video_path = os.path.join(outdir, f'{filename}.mp4')
        video = imageio.get_writer(
            video_path, mode='I', fps=fps, codec='libx264', bitrate='16M'
            )
        print(f'Saving optimization progress video "{video_path}"')
        for projected_w in tqdm(
            projected_w_steps, total=len(projected_w_steps)
            ):
            synth_image = G.synthesis(
                projected_w.unsqueeze(0), noise_mode='const'
                )
            synth_image = (synth_image + 1) * (255 / 2)
            synth_image = synth_image.permute(0, 2, 3, 1).clamp(0, 255).to(
                torch.uint8
                )[0].cpu().numpy()
            video.append_data(synth_image)
        video.close()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    run_projection()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------