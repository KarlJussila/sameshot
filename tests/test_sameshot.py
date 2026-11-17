from pathlib import Path
import shutil

from PIL import Image, ImageDraw, ImageEnhance, ImageOps
import pytest

from sameshot import compare, scan
from sameshot.align import measure
from sameshot.store import file_hash
from sameshot.cli import main

IMAGES = Path(__file__).parent / 'images'
PHOTOS = sorted(IMAGES.glob('*.jpg'))


def open_photo(name):
    return Image.open(IMAGES / f'{name}.jpg').convert('RGB')


def crop(image):
    width, height = image.size
    return image.crop((width // 25, height // 25, width - width // 25, height - height // 25))


def watermark(image):
    image = image.copy()
    draw = ImageDraw.Draw(image, 'RGBA')
    width, height = image.size
    draw.rectangle((width * .55, height * .85, width * .97, height * .97), fill=(0, 0, 0, 140))
    draw.text((width * .58, height * .88), '@somebody', fill=(255, 255, 255, 230), font_size=max(12, height // 20))
    return image


def moved_subject(image):
    """The same background with a large region shifted, standing in for a second shot."""
    image = image.copy()
    width, height = image.size
    box = (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
    image.paste(image.crop(box), (box[0] + width // 12, box[1] + height // 20))
    return image


VARIANTS = {
    'crop': crop,
    'downscale': lambda image: image.resize((image.width * 2 // 3, image.height * 2 // 3), Image.Resampling.LANCZOS),
    'mirror': ImageOps.mirror,
    'watermark': watermark,
    'brighter': lambda image: ImageEnhance.Contrast(ImageEnhance.Brightness(image).enhance(1.2)).enhance(1.15),
    'half': lambda image: image.resize((image.width // 2, image.height // 2), Image.Resampling.LANCZOS),
    'everything': lambda image: watermark(ImageOps.mirror(crop(image))).resize((image.width // 2, image.height // 2)),
}


def save(image, path, quality=90):
    image.save(path, quality=quality)
    return path


@pytest.fixture
def cache(tmp_path):
    return tmp_path / 'cache'


@pytest.mark.parametrize('variant', VARIANTS)
def test_edited_copies_are_the_same_photo(tmp_path, cache, variant):
    original = save(open_photo('coffee'), tmp_path / 'original.jpg')
    edited = save(VARIANTS[variant](open_photo('coffee')), tmp_path / 'edited.jpg', quality=80)
    assert compare(original, edited, cache_dir=cache)['same']


def test_different_photos_and_shots_are_not(tmp_path, cache):
    for a, b in [('coffee', 'chelsea'), ('astronaut', 'rocket')]:
        assert not compare(IMAGES / f'{a}.jpg', IMAGES / f'{b}.jpg', cache_dir=cache)['same']
    moved = save(moved_subject(open_photo('astronaut')), tmp_path / 'moved.jpg')
    assert not compare(IMAGES / 'astronaut.jpg', moved, cache_dir=cache)['same']


def test_tight_crops_match_whichever_file_comes_first(tmp_path, cache):
    photo = open_photo('astronaut')
    width, height = photo.size
    tight = save(photo.crop((width // 3, height // 4, width * 2 // 3, height * 3 // 4)), tmp_path / 'tight.jpg')
    pair = [(file_hash(path), str(path)) for path in (IMAGES / 'astronaut.jpg', tight)]
    for left, right in (pair, pair[::-1]):
        options = measure(cache, left, right)
        assert any(option['overlap'] >= .95 and option['matching_tiles'] >= .75 for option in options)


def test_collage_matches_each_photo_in_it(tmp_path, cache):
    left, right = open_photo('coffee'), open_photo('chelsea')
    right = right.resize((right.width * left.height // right.height, left.height))
    collage = Image.new('RGB', (left.width + right.width, left.height))
    collage.paste(left, (0, 0))
    collage.paste(right, (left.width, 0))
    collage = save(collage, tmp_path / 'collage.jpg')
    assert compare(IMAGES / 'coffee.jpg', collage, cache_dir=cache)['same']
    assert compare(IMAGES / 'chelsea.jpg', collage, cache_dir=cache)['same']


def test_identical_files(tmp_path, cache):
    copy = shutil.copy(IMAGES / 'rocket.jpg', tmp_path / 'rocket.jpg')
    assert compare(IMAGES / 'rocket.jpg', copy, cache_dir=cache)['identical']


def test_scan_groups_copies(tmp_path, cache):
    folder = tmp_path / 'photos'
    (folder / 'nested').mkdir(parents=True)
    for photo in PHOTOS:
        shutil.copy(photo, folder / photo.name)
    for name, variant in VARIANTS.items():
        save(variant(open_photo('astronaut')), folder / 'nested' / f'astronaut-{name}.jpg', quality=80)
    save(moved_subject(open_photo('coffee')), folder / 'coffee-moved.jpg')
    shutil.copy(IMAGES / 'rocket.jpg', folder / 'nested' / 'rocket-copy.jpg')
    (folder / 'broken.jpg').write_bytes(b'not an image')

    report = scan([folder], cache_dir=cache, workers=2, log=lambda message: None)
    groups = sorted(sorted(Path(file['path']).name for file in group['files']) for group in report['groups'])
    assert groups == [sorted(['astronaut.jpg', *(f'astronaut-{name}.jpg' for name in VARIANTS)]),
                      ['rocket-copy.jpg', 'rocket.jpg']]
    assert [Path(error['path']).name for error in report['errors']] == ['broken.jpg']

    again = scan([folder], cache_dir=cache, workers=2, log=lambda message: None)
    assert again['groups'] == report['groups']


def test_cli_compare_exit_code(cache, capsys):
    assert main(['compare', str(IMAGES / 'coffee.jpg'), str(IMAGES / 'chelsea.jpg'), '--cache', str(cache)]) == 1


def test_shrinking_a_larger_copy_keeps_pixel_alignment():
    from sameshot.align import matched_resolution, tile_correlations
    import cv2
    import numpy as np
    small = np.asarray(open_photo('astronaut'))
    large = cv2.resize(small, (small.shape[1] * 3 // 2, small.shape[0] * 3 // 2), interpolation=cv2.INTER_LINEAR)
    matrix = np.array([[small.shape[1] / large.shape[1], 0, .5 * (small.shape[1] / large.shape[1] - 1)],
                       [0, small.shape[0] / large.shape[0], .5 * (small.shape[0] / large.shape[0] - 1)]])
    other, reference, warp = matched_resolution(large, small, matrix)
    warped = cv2.warpAffine(other, warp, small.shape[1::-1])
    inside = np.zeros(small.shape[:2], bool)
    inside[8:-8, 8:-8] = True
    assert min(tile_correlations(reference, warped, inside)) >= .99
