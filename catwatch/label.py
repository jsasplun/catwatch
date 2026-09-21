# Author: John Asplund
# Date Created: 9/20/26
# AI tool: Claude Opus 5

"""Label captured images with the keyboard. See the labeling guide in the
README.

    python -m catwatch.label --labeled-by sora
"""

from __future__ import annotations

import argparse
import getpass

from catwatch.settings import bowl_crop_box, load_config, project_path
from core.cv_tools.capture_store import read_capture_log
from core.cv_tools.labeler import KeyboardLabeler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labeled-by",
        default=getpass.getuser(),
        help="Name stored with each label,"
        + " useful if more than one person labels.",
    )
    args = parser.parse_args()

    config = load_config()
    raw_dir = project_path(config["paths"]["raw_dir"])
    image_paths = [row["image_path"] for row in read_capture_log(raw_dir)]
    labeler = KeyboardLabeler(
        image_root=raw_dir,
        labels_file=project_path(config["paths"]["labels_file"]),
        key_to_label=config["label_keys"],
        labeled_by=args.labeled_by,
        highlight_box=bowl_crop_box(config),
    )
    count = labeler.run(image_paths)
    print(f"Wrote {count} labels this session.")


if __name__ == "__main__":
    main()
