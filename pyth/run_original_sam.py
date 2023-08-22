import getopt
import os
import sys
import time
import requests
import numpy as np
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor

from utils import cache_dir, get_checkpoint_url, get_checkpoint_path


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def fetch_image_encoder(checkpoint_name: str):
    checkpoint_path = get_checkpoint_path(checkpoint_name)
    checkpoint_url = get_checkpoint_url(checkpoint_name)

    # creating cache directory and downloading the model checkpoint
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(checkpoint_path):
        req = requests.get(checkpoint_url, allow_redirects=True)
        with open(checkpoint_path, "wb") as cp_file:
            cp_file.write(req.content)

    return checkpoint_path


def print_help():
    print("python run_original_sam.py [OPTIONS] image_path points labels")
    print("Run original segment-anything")
    print("Mandatory arguments:")
    print("\t image_path: The path to image to process, e.g. \"..\\data\\test_image.jpg\"")
    print("\t points: an array of 2d coordinates for segmentation, e.g. \"926, 926, 806, 918\"")
    print("\t         Each pair of adjacent comma-separated values is treated as a 2d-point")
    print("\t labels: Labels corresponding to the provided 2d coordinates, e.g. \"1, 0\"")
    print("\t         The number of labels must coincide with the number of input points")
    print("Optional arguments:")
    print("\t-h, --help: print this help message and exit without processing")
    print("\t-c, --checkpoint: VIT weights checkpoint to download.")
    print("\t                  Possible values are vit_b (default), vit_l, vit_h.")

    sys.exit()


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hci:", ["help", "checkpoint=", "image_path="])

    image_path = str()
    checkpoint_name = "vit_b"
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
        elif opt in ("-c", "--checkpoint"):
            checkpoint_name = arg

    image_path = None if len(args) < 1 else args[0]
    if image_path is None:
        raise RuntimeError("Image path was not provided, you need to provide at least image path, points and labels,\n"
                           "run run_original_sam.py -h for full info.")
    if not (os.path.exists(image_path) and os.path.isfile(image_path)):
        raise RuntimeError(f"{image_path} does not exist")

    input_points = None if len(args) < 2 else np.fromstring(args[1], sep=',').reshape((-1, 2))
    if input_points is None:
        raise RuntimeError("Input points for segmentation are not defined,\n"
                           "run run_original_sam.py -h for full info.")
    input_labels = None if len(args) < 3 else np.fromstring(args[2], sep=',')
    if input_labels is None:
        raise RuntimeError("Input labels for segmentation are not defined,\n"
                           "run run_original_sam.py -h for full info.")
    if len(input_labels) != input_points.shape[0]:
        raise RuntimeError("Number of labels and points must be equal.")

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

    checkpoint_path = fetch_image_encoder(checkpoint_name)
    sam = sam_model_registry[checkpoint_name](checkpoint=checkpoint_path)
    predictor = SamPredictor(sam)

    t = time.perf_counter()
    predictor.set_image(image)
    print(f"Encoding input image in {time.perf_counter() - t} seconds")

    masks, scores, logits = predictor.predict(
        point_coords=input_points,
        point_labels=input_labels,
        multimask_output=False,
    )

    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # drawing the image with masking out the background
        ax1.imshow(mask[:, :, np.newaxis] * image)

        # drawing original image with input points
        ax2.imshow(image)
        show_points(input_points, input_labels, ax2)

        fig.suptitle(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
        ax1.axis('off')
        ax2.axis('off')
        plt.show()
