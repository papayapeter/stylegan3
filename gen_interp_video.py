# interpolation and combining ws from Derrick Schultz
# https://github.com/dvschultz/stylegan2-ada-pytorch
#
# to do:
# - implement interpolation between seeds (with truncation) and seed parsing through click (can be taken from gen_images.py)
# - implement parsing of projected_w files so 1-n files can be passed in
"""interpolate between two vectors in latent space and save the result as video"""

import os
from typing import Optional, Tuple

import click
from tqdm import tqdm
import dnnlib
import numpy as np
import imageio
import torch
import math

import legacy

#----------------------------------------------------------------------------


def make_transform(translate: Tuple[float, float], angle: float):
    m = np.eye(3)
    s = np.sin(angle / 360.0 * np.pi * 2)
    c = np.cos(angle / 360.0 * np.pi * 2)
    m[0][0] = c
    m[0][1] = s
    m[0][2] = translate[0]
    m[1][0] = -s
    m[1][1] = c
    m[1][2] = translate[1]
    return m


#----------------------------------------------------------------------------


def line_interpolate(endpoints, steps, easing):
    out = []
    for midpoint in range(len(endpoints) - 1):
        for index in range(int(steps)):
            t = index / float(steps)

            if (easing == 'linear'):
                out.append(
                    endpoints[midpoint + 1] * t
                    + endpoints[midpoint] * (1 - t)
                    )
            elif (easing == 'easeInOutQuad'):
                if (t < 0.5):
                    fr = 2 * t * t
                else:
                    fr = (-2 * t * t) + (4 * t) - 1
                out.append(
                    endpoints[midpoint + 1] * fr
                    + endpoints[midpoint] * (1 - fr)
                    )
            elif (easing == 'bounceEaseOut'):
                if (t < 4 / 11):
                    fr = 121 * t * t / 16
                elif (t < 8 / 11):
                    fr = (363 / 40.0 * t * t) - (99 / 10.0 * t) + 17 / 5.0
                elif t < 9 / 10:
                    fr = (4356 / 361.0 * t
                          * t) - (35442 / 1805.0 * t) + 16061 / 1805.0
                else:
                    fr = (54 / 5.0 * t * t) - (513 / 25.0 * t) + 268 / 25.0
                out.append(
                    endpoints[midpoint + 1] * fr
                    + endpoints[midpoint] * (1 - fr)
                    )
            elif (easing == 'circularEaseOut'):
                fr = np.sqrt((2 - t) * t)
                out.append(
                    endpoints[midpoint + 1] * fr
                    + endpoints[midpoint] * (1 - fr)
                    )
            elif (easing == 'circularEaseOut2'):
                fr = np.sqrt(np.sqrt((2 - t) * t))
                out.append(
                    endpoints[midpoint + 1] * fr
                    + endpoints[midpoint] * (1 - fr)
                    )
            elif (easing == 'backEaseOut'):
                p = 1 - t
                fr = 1 - (p * p * p - p * math.sin(p * math.pi))
                out.append(
                    endpoints[midpoint + 1] * fr
                    + endpoints[midpoint] * (1 - fr)
                    )

    return out


#----------------------------------------------------------------------------


# yapf: disable
@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--ws', 'projected_ws',     type=str, nargs=2, help='2 projected_w filenames to interpolate beween', required=True)
@click.option('--noise-mode',             help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--outdir',                 help='Where to save the output video', type=str, required=True, metavar='DIR')
@click.option('--name',                   help='Name of the output Video', type=str, required=True)
@click.option('--length',                 help='Length of the final video in seconds', type=float, default=10, show_default=True)
@click.option('--fps',                    help='Frames per second of final video', type=int, default=30, show_default=True)
# yapf: enable
def generate_interpolation(
    network_pkl: str,
    projected_ws: Tuple[str],
    noise_mode: str,
    outdir: str,
    name: str,
    length: float,
    fps: int,
    ):

    print(f'Loading networks from "{network_pkl}"...')
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)  # type: ignore

    os.makedirs(outdir, exist_ok=True)

    # load w from npz and combine
    ws = torch.tensor((), device=device)
    for f in projected_ws:
        w = torch.tensor(np.load(f)['w'], device=device)
        assert w.shape[1:] == (G.num_ws, G.w_dim)
        ws = torch.cat((ws, w), 0)

    # get interpolated points
    points = line_interpolate(ws, fps * length, 'easeInOutQuad')

    # generate video
    video = imageio.get_writer(
        os.path.join(outdir, f'{name}.mp4'),
        mode='I',
        fps=fps,
        codec='libx264',
        bitrate='16M'
        )
    for w in tqdm(points, total=len(points)):
        synth_image = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
        synth_image = (synth_image + 1) * (255 / 2)
        synth_image = synth_image.permute(0, 2, 3,
                                          1).clamp(0,
                                                   255).to(torch.uint8
                                                           )[0].cpu().numpy()
        video.append_data(synth_image)
    # backwards
    # for w in tqdm(reversed(points), total=len(points)):
    #     synth_image = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)
    #     synth_image = (synth_image + 1) * (255 / 2)
    #     synth_image = synth_image.permute(0, 2, 3,
    #                                       1).clamp(0,
    #                                                255).to(torch.uint8
    #                                                        )[0].cpu().numpy()
    #     video.append_data(synth_image)
    # video.close()


#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_interpolation()  # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
