# sameshot

Find copies of the same photo in a pile of images: the same original after
crops, resizing, recompression, mirroring, watermarks, and color edits. Burst
shots and other photos that merely look alike are kept apart.

```console
$ sameshot scan ~/Pictures/collection -o report.json
2381 images
SSCD: embedding 2381 images (0 cached)
Alignment: 914 candidate pairs, 914 to check
212 groups, 331 extra copies, 0 unreadable
```

## Why

Embedding models for near-duplicate detection give you a similarity score and
leave you to choose a threshold. Set it high and edited copies slip through;
set it low and different shots of the same scene get merged. For many
collections there is no threshold that does both. Perceptual hashes have the
same problem.

sameshot uses the model only to find candidates, with a deliberately low
threshold, and then decides each candidate with a geometric check: *do these
two images share the same pixels?*

1. **Candidates.** [SSCD](https://github.com/facebookresearch/sscd-copy-detection),
   a copy-detection model, embeds each image and its mirror. Pairs with cosine
   similarity of at least 0.70 are candidates.
2. **Alignment.** SIFT keypoints are matched, and RANSAC fits a single
   similarity transform (translation, rotation, and uniform scale), also with
   one image mirrored. The second image is warped onto the first.
3. **Verdict.** The image showing less of the scene (a crop, for example) is
   the reference. Both images are brought to the coarser of their two
   resolutions, and the reference's central 80% is cut into an 8×6 grid. The
   pair is the same photo when the other image covers at least 95% of that
   content and at least 75% of the tiles correlate at 0.99 or better.

A different shot fails step 3 even when it looks nearly identical, because the
subject moved, the camera moved, or both, and the pixels no longer line up
under one transform. An edited copy passes because crops, scaling, and
mirroring are exactly what the transform models. Watermarks, overlays, and
retouching only affect a few tiles, and color and contrast changes leave the
correlation intact.

In practice the verdict is close to exact. In our own use, checking a few
thousand candidate pairs by hand turned up four false positives and one false
negative, all from the limitations below.

## Limitations

What sameshot really detects is *the same capture*, so it fails where that
differs from *the same photo*:

- **Fixed camera, small subject.** A tripod or security camera shooting a
  distant subject produces distinct shots whose pixels are almost all
  unchanged background. They clear the tile check and are reported as the same
  photo. If the camera moves at all, or the subject fills a reasonable part of
  the frame, distinct shots fail the check as they should.
- **Edits across the whole image.** Effects that change every region, such as a
  heavy glow or stylized filter, together with large overlays, can push enough
  tiles below 0.99 that a real copy is missed.
- **Small, smooth images.** Sky, smoke, and other low-texture content at small
  sizes can fall below 0.99 after recompression alone. Larger images are
  downscaled to 1400px for the check, which averages most compression noise
  away.
- **Collages merge unrelated photos.** If a collage contains photos A and B,
  both A and B match the collage, so all three land in one group even
  though A and B are different photos. The `pairs` list shows which matches
  put a file in its group, so a merge like this can be found and split.

## Install

sameshot needs PyTorch. Install the build for your hardware first
([pytorch.org](https://pytorch.org/get-started/locally/)), then:

```console
pip install git+https://github.com/KarlJussila/sameshot
```

The SSCD weights (about 100 MB, MIT licensed) download into torch's hub
directory on first use. The embedding runs on the GPU when one is available;
alignment runs on CPU worker processes.

## Usage

```console
sameshot scan PATH... [-o report.json] [--threshold 0.70] [--min-overlap 0.95]
                      [--min-matching-tiles 0.75] [--device auto] [--workers N]
sameshot compare A B
```

`scan` takes files and folders (searched recursively) and writes a JSON report.
`compare` checks one pair directly, skipping the model, and exits 0 when the two
files are the same photo and 1 when they aren't.

Nothing is moved or deleted. The report is meant to be acted on by your own
script:

```json
{
  "images": 2381,
  "groups": [
    {
      "files": [
        {"path": "collection/IMG_2041.jpg", "sha256": "…", "width": 4032, "height": 3024},
        {"path": "collection/edited/2041-crop.jpg", "sha256": "…", "width": 1800, "height": 1350}
      ]
    }
  ],
  "pairs": [
    {
      "a": "collection/IMG_2041.jpg", "b": "collection/edited/2041-crop.jpg", "similarity": 0.8312, "same": true,
      "options": [{"mirror": false, "inliers": 412, "tiles": 48, "matching_tiles": 0.9792, "overlap": 1.0, "matrix": [[…], […]]}]
    }
  ],
  "errors": [{"path": "collection/broken.jpg", "error": "UnidentifiedImageError: …"}]
}
```

- `groups` lists each set of copies, including byte-identical files, with their
  sizes. Which copy to keep is up to you.
- `pairs` lists every candidate pair that was checked, matched or not, with
  the alignment measurements. Sort by `similarity` to review the borderline
  cases, or apply your own cutoffs to `overlap` and `matching_tiles`.

From Python:

```python
from sameshot import compare, scan

if __name__ == '__main__':  # alignment uses worker processes
    report = scan(['photos/'])
    print(compare('a.jpg', 'b.jpg')['same'])
```

The stages are importable on their own: `sameshot.embed` (SSCD embeddings and
candidate retrieval) and `sameshot.align` (features, the transform, and the
tile check).

## Cache

Everything sameshot computes is cached in `~/.cache/sameshot` (or `--cache`),
keyed by content hash and algorithm version: file hashes, embeddings, SIFT
features, and pair measurements. Re-running a scan after adding images only
computes what is new. Changing the overlap and tile cutoffs reuses the cached
measurements, and lowering the candidate threshold only checks the new pairs.
The cache can be deleted at any time.

## Tests

```console
pip install -e . pytest
pytest
```

The test photos are public domain or CC0, from
[scikit-image](https://scikit-image.org/docs/stable/api/skimage.data.html):
the astronaut Eileen Collins (NASA), coffee (Rachel Michetti), Chelsea the cat
(Stefan van der Walt), and a Falcon 9 launch (SpaceX).

## Credits

Candidate retrieval uses SSCD from Pizzi et al.,
[*A Self-Supervised Descriptor for Image Copy Detection*](https://arxiv.org/abs/2202.10261)
(CVPR 2022).

## License

MIT
