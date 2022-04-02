"""tool to align faces similarly to the ffhq dataset. These can then be used for transfer learning and projection"""

import os
import click
from glob import glob

from alignement.align_face import align_face
from dataset_tool import error

@click.command()
@click.option('--predictor', 'predictor_dat', help='Landmark detection model filename', required=True, metavar='PATH')
@click.option('--source', help='Directory for input png images', required=True, metavar='PATH')
@click.option('--dest', help='Output directory for aligned images', required=True, metavar='PATH')
def run_alignment(predictor_dat, source, dest):
    # create output directory, if it does not exists
    os.makedirs(dest, exist_ok=True)

    # check source directory
    if not os.path.isdir(source):
        error('source is not a directory')

    # find all images in source directory
    if not len(matches := glob(os.path.join(source, '*.png'))):
        error(f'no .png images found in {source}')

    for path in matches:
        align_face(path, predictor_dat).save(os.path.join(dest, os.path.basename(path)))
        

if __name__ == "__main__":
    run_alignment() # pylint: disable=no-value-for-parameter