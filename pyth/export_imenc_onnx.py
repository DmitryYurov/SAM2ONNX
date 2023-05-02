import warnings
import torch
from torch import nn as nn
import onnxruntime

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam


class ExportImageEncoder(nn.Module):
    def __init__(self, model: Sam) -> None:
        super().__init__()
        self.image_encoder = model.image_encoder
        self.image_size = model.image_encoder.img_size

    @torch.no_grad()
    def forward(self, input_image: torch.Tensor):
        return self.image_encoder(input_image)


if __name__ == "__main__":
    export_path = "../export/image_encoder.onnx"
    sam = sam_model_registry["vit_b"](checkpoint="../data/sam_vit_b_01ec64.pth")
    to_onnx = ExportImageEncoder(sam)
    dummy_inputs = {
        "input_image": torch.normal(mean=0.0, std=1.0, size=(1, 3, to_onnx.image_size, to_onnx.image_size))
    }
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=torch.jit.TracerWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        with open(export_path, "wb") as f:
            torch.onnx.export(
                to_onnx,
                tuple(dummy_inputs.values()),
                f,
                export_params=True,
                verbose=False,
                opset_version=17,
                do_constant_folding=True,
                input_names=list(dummy_inputs.keys()),
            )

    onnx_session = onnxruntime.InferenceSession(export_path)
    onnx_sess_inputs = {
        "input_image": torch.normal(mean=0.0, std=1.0,
                                    size=(1, 3, to_onnx.image_size, to_onnx.image_size)).cpu().detach().numpy()
    }
    outputs = onnx_session.run(None, onnx_sess_inputs)
    if len(outputs) == 1 and outputs[0].shape == (1, 256, 64, 64):
        print("Successfully exported the image encoder")
