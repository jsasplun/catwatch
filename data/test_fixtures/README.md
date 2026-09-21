# Test fixtures

Synthetic (drawn by code, not photographed) images and video for the tests.
They are NOT training data and are kept apart from data/raw/ and
data/labels.csv on purpose. Do not label, edit, or add to them: tests check
them against MANIFEST.json.

- raw/stills/*.jpg + raw/captures.csv + labels.csv: 14 images in the same
  formats as the real dataset. Each split (train/val/test) holds one image of
  each class, plus two `unusable` images.
- bowl_visits.avi + bowl_visits.json: a 20 second video. Black-patched cat
  visits, then the orange-patched cat. The json lists the exact frames.

Regenerate into a NEW folder with `python -m tests.make_fixtures --output <dir>`.
