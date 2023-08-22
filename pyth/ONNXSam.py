import os
import sys
from typing import Tuple
import time
import getopt
from copy import deepcopy

import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch
from torch.nn import functional as Func
from torchvision.transforms.functional import resize, to_pil_image  # type: ignore

import onnxruntime


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


class ResizeLongestSide:
    """
    Resizes images to longest side 'target_length', as well as provides
    methods for resizing coordinates and boxes. Provides methods for
    transforming both numpy array and batched torch tensors.
    """

    def __init__(self, target_length: int) -> None:
        self.target_length = target_length

    def apply_image(self, image: np.ndarray) -> np.ndarray:
        """
        Expects a numpy array with shape HxWxC in uint8 format.
        """
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return np.array(resize(to_pil_image(image), target_size))

    def apply_coords(self, coords: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array of length 2 in the final dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).astype(float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes(self, boxes: np.ndarray, original_size: Tuple[int, ...]) -> np.ndarray:
        """
        Expects a numpy array shape Bx4. Requires the original image size
        in (H, W) format.
        """
        boxes = self.apply_coords(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    def apply_image_torch(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects batched images with shape BxCxHxW and float format. This
        transformation may not exactly match apply_image. apply_image is
        the transformation expected by the model.
        """
        # Expects an image in BCHW format. May not exactly match apply_image.
        target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.target_length)
        return Func.interpolate(
            image, target_size, mode="bilinear", align_corners=False, antialias=True
        )

    def apply_coords_torch(
        self, coords: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with length 2 in the last dimension. Requires the
        original image size in (H, W) format.
        """
        old_h, old_w = original_size
        new_h, new_w = self.get_preprocess_shape(
            original_size[0], original_size[1], self.target_length
        )
        coords = deepcopy(coords).to(torch.float)
        coords[..., 0] = coords[..., 0] * (new_w / old_w)
        coords[..., 1] = coords[..., 1] * (new_h / old_h)
        return coords

    def apply_boxes_torch(
        self, boxes: torch.Tensor, original_size: Tuple[int, ...]
    ) -> torch.Tensor:
        """
        Expects a torch tensor with shape Bx4. Requires the original image
        size in (H, W) format.
        """
        boxes = self.apply_coords_torch(boxes.reshape(-1, 2, 2), original_size)
        return boxes.reshape(-1, 4)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)


class ONNXSam:
    pixel_mean = torch.Tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
    pixel_std = torch.Tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)
    input_size = 1024
    mask_threshold: float = 0.0

    def __init__(self, imenc_path: str, therest_path: str):
        if not (os.path.exists(imenc_path) and os.path.isfile(imenc_path)):
            raise RuntimeError(f"Image encoder coudn't be found at the path {imenc_path}")
        if not (os.path.exists(therest_path) and os.path.isfile(therest_path)):
            raise RuntimeError(f"SAM back-part coudn't be found at the path {therest_path}")
        self.session_imenc = onnxruntime.InferenceSession(imenc_path)
        self.session_therest = onnxruntime.InferenceSession(therest_path)

    def _preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.input_size - h
        padw = self.input_size - w
        x = Func.pad(x, (0, padw, 0, padh))
        return x

    def preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        input_image = ResizeLongestSide(self.input_size).apply_image(image)
        input_image_torch = torch.as_tensor(input_image)
        input_image_torch = input_image_torch.permute(2, 0, 1).contiguous()[None, :, :, :]

        original_size = np.array(image.shape[:2])
        return self._preprocess(input_image_torch).cpu().detach().numpy(), original_size

    @staticmethod
    def resize_longest_image_size(
            input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def mask_postprocessing(self, masks: torch.Tensor, orig_im_size: torch.Tensor) -> torch.Tensor:
        masks = Func.interpolate(
            masks,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_im_size, self.input_size)
        masks = masks[..., : int(prepadded_size[0]), : int(prepadded_size[1])]

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = Func.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def predict(self, image: np.ndarray, points: np.ndarray, point_labels: np.ndarray):
        processed_image, original_size = self.preprocess_image(image)
        features = self.session_imenc.run(None, {"input_image": processed_image})

        onnx_coord = points[None, :, :]
        onnx_coord = ResizeLongestSide(self.input_size).apply_coords(onnx_coord, image.shape[:2]).astype(np.float32)
        onnx_label = point_labels[None, :].astype(np.float32)

        onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
        onnx_has_mask_input = np.zeros(1, dtype=np.float32)

        ort_inputs = {
            "image_embeddings": features[0],
            "point_coords": onnx_coord,
            "point_labels": onnx_label,
            "mask_input": onnx_mask_input,
            "has_mask_input": onnx_has_mask_input
        }
        masks, scores = self.session_therest.run(None, ort_inputs)
        masks = self.mask_postprocessing(torch.from_numpy(masks),
                                         torch.from_numpy(original_size)).cpu().detach().numpy()

        return masks > self.mask_threshold, scores


def print_help():
    print("python ONNXSam.py [OPTIONS] export_path image_path points labels")
    print("Run ONNX-version of the segment-anything")
    print("Mandatory arguments:")
    print("\t export_path: The path to onnx-exported SAM, e.g. \".\\export\"")
    print("\t image_path: The path to image to process, e.g. \"..\\data\\test_image.jpg\"")
    print("\t points: an array of 2d coordinates for segmentation, e.g. \"926, 926, 806, 918, 0, 0\"")
    print("\t         Each pair of adjacent comma-separated values is treated as a 2d-point")
    print("\t labels: Labels corresponding to the provided 2d coordinates, e.g. \"1, 0, -1\"")
    print("\t         The number of labels must coincide with the number of input points")
    print("\t Note that points and labels require additional padding input ((0, 0) and -1 respectively) "
          "as described in the original script for onnx-export at "
          "https://github.com/facebookresearch/segment-anything/blob/main/notebooks/onnx_model_example.ipynb "
          "in case there is no box input for the model")
    print("Optional arguments:")
    print("\t-h, --help: print this help message and exit without processing")

    sys.exit()


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()

    export_path = None if len(args) < 1 else args[0]
    if export_path is None or not os.path.exists(export_path) or not os.path.isdir(export_path):
        raise RuntimeError("Export path is not defined, doesn't exist or is not a directory,\n"
                           "run ONNXSam.py -h for full info.")

    predictor = ONNXSam(export_path + os.path.sep + "image_encoder.onnx", export_path + os.path.sep + "the_rest.onnx")

    image_path = None if len(args) < 2 else args[1]
    if image_path is None:
        raise RuntimeError("Image path was not provided, you need to provide at least image path, points and labels,\n"
                           "run ONNXSam.py -h for full info.")
    if not (os.path.exists(image_path) and os.path.isfile(image_path)):
        raise RuntimeError(f"{image_path} does not exist")

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_points = None if len(args) < 2 else np.fromstring(args[2], sep=',').reshape((-1, 2))
    if input_points is None:
        raise RuntimeError("Input points for segmentation are not defined,\n"
                           "run ONNXSam.py -h for full info.")
    input_labels = None if len(args) < 3 else np.fromstring(args[3], sep=',')
    if input_labels is None:
        raise RuntimeError("Input labels for segmentation are not defined,\n"
                           "run ONNXSam.py -h for full info.")
    if len(input_labels) != input_points.shape[0]:
        raise RuntimeError("Number of labels and points must be equal.")

    t = time.perf_counter()
    masks, scores = predictor.predict(image, input_points, input_labels)
    print(f"Prediction time: {time.perf_counter() - t} seconds")

    for i, (mask, score) in enumerate(zip(masks, scores)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # drawing the image with masking out the background
        ax1.imshow(mask[0, :, :, np.newaxis] * image)

        # drawing original image with input points
        ax2.imshow(image)
        show_points(input_points, input_labels, ax2)

        fig.suptitle(f"Mask {i + 1}, Score: {score[0]:.3f}", fontsize=18)
        ax1.axis('off')
        ax2.axis('off')
        plt.show()
