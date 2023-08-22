import os

cache_dir = "." + os.sep + "cache"


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