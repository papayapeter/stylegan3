"""tool to align faces similarly to the ffhq dataset. These can then be used for transfer learning and projection"""

import os
import click
from pathlib import Path
import PIL.Image

from alignement.align_face import align_face
from dataset_tool import error, file_ext, is_image_ext


# yapf: disable
@click.command()
@click.option('--predictor', 'predictor_dat', help='Landmark detection model filename', required=True, metavar='PATH')
@click.option('--source',                     type=click.Path(exists=True, file_okay=False), help='Directory for input images', required=True, metavar='PATH')
@click.option('--dest',                       type=click.Path(file_okay=False), help='Output directory for aligned images', required=True, metavar='PATH')
# yapf: enable
def run_alignment(predictor_dat, source, dest):
    PIL.Image.init()

    # create output directory, if it does not exists
    os.makedirs(dest, exist_ok=True)

    # find all images in source directory
    if not len(
        matches := [
            str(f) for f in sorted(Path(source).rglob('*'))
            if is_image_ext(f) and os.path.isfile(f)
            ]
        ):
        error(f'no compatible images found in {source}')

    # align faces and save files for the number of matches
    for path in matches:
        if (imgs := align_face(path, predictor_dat)) is not None:
            if len(imgs) == 1:
                imgs[0].save(
                    os.path.join(
                        dest,
                        os.path.basename(path).replace(
                            file_ext(os.path.basename(path)), 'png'
                            )
                        )
                    )

            else:
                for index, img in enumerate(imgs):
                    name, extension = os.path.splitext(os.path.basename(path))
                    img.save(os.path.join(dest, f'{name}_{index:02d}.png'))


if __name__ == "__main__":
    run_alignment()  # pylint: disable=no-value-for-parameter
