from rembg import remove
from PIL import Image
import torch
import numpy as np
from transparent_background import Remover
import folder_paths as comfy_paths
import os
MODELS_DIR =  comfy_paths.models_dir
rembg_dir = os.path.join(MODELS_DIR, 'rembg')
model_dir = rembg_dir+os.sep+'latest_test.pth'
remover = Remover(ckpt=model_dir) # default setting

# Tensor to PIL
def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

# Convert PIL to Tensor
def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

class ImageRemoveBackgroundRembg:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_background"
    CATEGORY = "image"

    def remove_background(self, image):
        image = tensor2pil(image)
        out = pil2tensor(remover.process(image, type='white',threshold=0.1))
        return (out,)


# A dictionary that contains all nodes you want to export with their names
# NOTE: names should be globally unique
NODE_CLASS_MAPPINGS = {
    "Image Remove Background (rembg)": ImageRemoveBackgroundRembg
}
