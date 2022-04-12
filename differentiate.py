"""tool for finding similar images (pixelwise) and discarding one, so only different images are being kept. useful for datasets from video"""

import os
import click
import shutil
import numpy as np
from glob import glob
from PIL import Image, ImageChops

from dataset_tool import error


# yapf: disable
@click.command()
@click.option('--source',                             type=click.Path(exists=True, file_okay=False), help='Directory or archive name for input dataset', required=True, metavar='PATH')
@click.option('--dest',                               type=click.Path(file_okay=False), help='Output directory or archive name for output dataset', required=True, metavar='PATH')
@click.option('--min-deviation', 'min_avg_deviation', help='Minimum average difference by pixel', metavar='FLOAT', type=click.FloatRange(min=0), default=2.0, show_default=True, required=True)
# yapf: enable
def differentiate(source: str, dest: str, min_avg_deviation: float):
    # find all images in source directory
    if not len(image_paths := sorted(glob(os.path.join(source, '*.png')))):
        error(f'no .png images found in {source}')

    # create output directory, if it does not exists
    os.makedirs(dest, exist_ok=True)

    while (remaining := len(image_paths)) > 0:
        # loading image
        image_path = image_paths.pop(0)
        base = np.array(Image.open(image_path).convert('L'))
        w, h = base.shape

        # copying image to dest directory
        new_path = shutil.copy2(image_path, dest)
        print(
            f'copied {image_path} to {new_path}. images remaining: {remaining}'
            )

        # comparing image
        immuted_paths = image_paths.copy()
        for path in immuted_paths:
            comparision = np.array(Image.open(path).convert('L'))
            difference = base.astype(np.int16) - comparision.astype(np.int16)
            abs_difference = abs(difference)
            avg_difference = np.sum(abs_difference) / (w * h)

            if avg_difference < min_avg_deviation:
                image_paths.remove(path)


if __name__ == '__main__':
    differentiate()