#!/usr/bin/env python3

import getopt
import os
import sys
import warnings
from typing import Tuple

import requests

import torch

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam

cache_dir = "." + os.sep + "cache"


class ImageEncoderExp(torch.nn.Module):
    def __init__(self, model: Sam) -> None:
        super().__init__()
        self.image_encoder = model.image_encoder
        self.image_size = model.image_encoder.img_size

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        return self.image_encoder(input_image)


class ExportTheRest(torch.nn.Module):
    def __init__(self, model: Sam) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size

    def _embed_points(self, point_coords: torch.Tensor, point_labels: torch.Tensor) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = point_embedding + self.model.prompt_encoder.not_a_point_embed.weight * (
            point_labels == -1
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = point_embedding + self.model.prompt_encoder.point_embeddings[
                i
            ].weight * (point_labels == i)

        return point_embedding

    def _embed_masks(self, input_mask: torch.Tensor, has_mask_input: torch.Tensor) -> torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(input_mask)
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), best_idx, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(self,
                image_embeddings: torch.Tensor,
                point_coords: torch.Tensor, point_labels: torch.Tensor,
                mask_input: torch.Tensor, has_mask_input: torch.Tensor):
        sparse_embedding = self._embed_points(point_coords, point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding
        )

        # the next line selects the best mask, equiv. return_single_mask=True for SamOnnxModel
        masks, scores = self.select_masks(masks, scores, point_coords.shape[1])
        return masks, scores


def get_checkpoint_url(checkpoint_name: str):
    if checkpoint_name == "vit_b":
        return "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
    elif checkpoint_name == "vit_h":
        return "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
    elif checkpoint_name == "vit_l":
        return "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth"
    else:
        raise RuntimeError(f"Given checkpoint {checkpoint_name} is unknown")


def get_checkpoint_path(checkpoint_name: str):
    if checkpoint_name == "vit_b":
        return cache_dir + os.sep + "sam_vit_b_01ec64.pth"
    elif checkpoint_name == "vit_h":
        return cache_dir + os.sep + "sam_vit_h_4b8939.pth"
    elif checkpoint_name == "vit_l":
        return cache_dir + os.sep + "sam_vit_l_0b3195.pth"
    else:
        raise RuntimeError(f"Given checkpoint {checkpoint_name} is unknown")


def export_im_encoder(export_dir: str, checkpoint_name: str):
    checkpoint_path = get_checkpoint_path(checkpoint_name)
    checkpoint_url = get_checkpoint_url(checkpoint_name)

    # creating cache directory and downloading the model checkpoint
    os.makedirs(cache_dir, exist_ok=True)
    if not os.path.exists(checkpoint_path):
        req = requests.get(checkpoint_url, allow_redirects=True)
        with open(checkpoint_path, "wb") as cp_file:
            cp_file.write(req.content)

    # creating export directory
    os.makedirs(export_dir, exist_ok=True)
    export_path = export_dir + os.sep + "image_encoder.onnx"

    # initializing export model
    sam = sam_model_registry[checkpoint_name](checkpoint=checkpoint_path)
    to_onnx = ImageEncoderExp(sam)

    # doing export
    dummy_inputs = {
        "input_image": torch.normal(mean=0.0, std=1.0, size=(1, 3, to_onnx.image_size, to_onnx.image_size))
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(export_path, "wb") as export_f:
            torch.onnx.export(
                to_onnx,
                tuple(dummy_inputs.values()),
                export_f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
            )


def export_the_rest(export_dir: str, checkpoint_name: str):
    # by the moment of calling this function the export path as well as the saved weights' checkpoint
    # should already exist
    export_path = export_dir + os.sep + "the_rest.onnx"
    sam = sam_model_registry[checkpoint_name](get_checkpoint_path(checkpoint_name))
    to_onnx = ExportTheRest(sam)

    dynamic_axes = {
        "point_coords": {1: "num_points"},
        "point_labels": {1: "num_points"},
    }

    embed_dim = sam.prompt_encoder.embed_dim
    embed_size = sam.prompt_encoder.image_embedding_size
    mask_input_size = [4 * x for x in embed_size]
    dummy_inputs = {
        "image_embeddings": torch.randn(1, embed_dim, *embed_size, dtype=torch.float),
        "point_coords": torch.randint(low=0, high=1024, size=(1, 5, 2), dtype=torch.float),
        "point_labels": torch.randint(low=0, high=4, size=(1, 5), dtype=torch.float),
        "mask_input": torch.randn(1, 1, *mask_input_size, dtype=torch.float),
        "has_mask_input": torch.tensor([1], dtype=torch.float)
    }
    output_names = ["masks", "iou_predictions"]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(export_path, "wb") as export_f:
            torch.onnx.export(
                to_onnx,
                tuple(dummy_inputs.values()),
                export_f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
                output_names=output_names,
                dynamic_axes=dynamic_axes
            )


def print_help():
    print("python export_model.py [OPTIONS] export_directory")
    print("Export segment-anything to ONNX format and save output in <export_directory>")
    print("Two files with names \"image_encoder.onnx\" and \"the_rest.onnx\" will be created.")
    print("Optional arguments:")
    print("\t-h, --help: print this help message and exit without processing")
    print("\t-c, --checkpoint: VIT weights checkpoint to download.")
    print("\t                  Possible values are vit_b (default), vit_l, vit_h.")


if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hc:", ["help", "checkpoint="])
    checkpoint_name = "vit_b"
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print_help()
            sys.exit()
        elif opt in ("-c", "--checkpoint"):
            checkpoint_name = arg

    export_directory = None if len(args) == 0 else args[0]
    if export_directory is None:
        print_help()
        raise RuntimeError("Export directory is not specified")

    export_im_encoder(export_directory, checkpoint_name)
    export_the_rest(export_directory, checkpoint_name)
