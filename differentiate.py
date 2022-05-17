"""tool for finding similar images (pixelwise) and discarding one, so only different images are being kept. useful for datasets from video"""

import os
import click
import shutil
from tqdm import tqdm
import numpy as np
from glob import glob
from PIL import Image
import random

from dataset_tool import error


# yapf: disable
@click.command()
@click.option('--source',                             type=click.Path(exists=True, file_okay=False), help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest',                               type=click.Path(file_okay=False), help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--min-deviation', 'min_avg_deviation', help='Minimum average difference by pixel', metavar='FLOAT', type=click.FloatRange(min=0), default=2.0, show_default=True, required=True)
@click.option('--batchsize', help='how many images in one batch', metavar='INT', type=click.IntRange(min=0), default=1000, show_default=True, required=True)
# @click.option('--batches', help='how many images in one batch', metavar='INT', type=click.IntRange(min=0), default=1, show_default=True, required=True)
# yapf: enable
def differentiate(
        source: str,
        dest: str,
        min_avg_deviation: float,
        batchsize: int,  # batches: int
    ):
    # find all images in source directory
    if not len(image_paths := sorted(glob(os.path.join(source, '*.png')))):
        error(f'no .png images found in {source}')

    print(f'{len(image_paths)} images found')

    # compare images in batches in multiple passes
    passes = 0
    while True:
        # if remaining images are smaller than batchsize, execute one last time
        last_pass = False
        if (len(image_paths)) < batchsize:
            last_pass = True

        passes += 1
        print(
            f'executing pass {passes}{" . this is the last pass" if last_pass else ""}'
            )

        immuted_image_paths = image_paths.copy()

        # shuffle the list after the first pass (making stagnation less likely)
        # leave it ordered on the first pass, because similar images are likely closer together
        if passes > 0:
            random.shuffle(immuted_image_paths)

        total_dicarded = 0
        for i in range(0, len(image_paths), batchsize):
            image_paths_chunk = immuted_image_paths[i:i + batchsize]

            print(f'processing images {i} to {i + len(image_paths_chunk) -1}')

            # load images into memory
            print('loading images')

            images_chunk = {}
            for image_path in tqdm(image_paths_chunk):
                images_chunk[image_path] = np.array(
                    Image.open(image_path).convert('L')
                    )

            print('comparing images')

            discarded = 0
            while (len(images_chunk)):
                # get base for comparison
                base = images_chunk.pop(list(images_chunk.keys())[0])
                w, h = base.shape

                # comparing image
                immuted_chunk_paths = list(images_chunk.keys())
                for path in immuted_chunk_paths:
                    comparision = images_chunk[path]
                    difference = base.astype(np.int16
                                             ) - comparision.astype(np.int16)
                    abs_difference = abs(difference)
                    avg_difference = np.sum(abs_difference) / (w * h)

                    # remove path from chunk and all image paths, if it's to similar
                    if avg_difference < min_avg_deviation:
                        images_chunk.pop(path)
                        image_paths.remove(path)

                        discarded += 1

                print(
                    f'discarded {discarded} images from chunk in total. {len(images_chunk)} remaining for comparison'
                    )

            total_dicarded += discarded

        print(f'{len(image_paths)} images remaining in total')

        if total_dicarded == 0:
            print(
                f'warning: comparison has stagnated! stopping here. execution with bigger batchsize recommended!'
                )
            break

        if last_pass: break

    print('copying images over to destination directory')

    # create output directory, if it does not exists
    os.makedirs(dest, exist_ok=True)

    for path in tqdm(image_paths):
        shutil.copy2(path, dest)

    # --- ORIGINAL CODE ---

    # # create output directory, if it does not exists
    # os.makedirs(dest, exist_ok=True)

    # while (remaining := len(image_paths)) > 0:
    #     # loading image
    #     image_path = image_paths.pop(0)
    #     base = np.array(Image.open(image_path).convert('L'))
    #     w, h = base.shape

    #     # copying image to dest directory
    #     new_path = shutil.copy2(image_path, dest)
    #     print(
    #         f'copied {image_path} to {new_path}. images remaining: {remaining}'
    #         )

    #     # comparing image
    #     immuted_paths = image_paths.copy()
    #     discarded = 0
    #     for path in tqdm.tqdm(immuted_paths, total=len(immuted_paths)):
    #         comparision = np.array(Image.open(path).convert('L'))
    #         difference = base.astype(np.int16) - comparision.astype(np.int16)
    #         abs_difference = abs(difference)
    #         avg_difference = np.sum(abs_difference) / (w * h)

    #         if avg_difference < min_avg_deviation:
    #             image_paths.remove(path)
    #             discarded += 1

    #     print(f'discarded {discarded} images')


if __name__ == '__main__':
    differentiate()