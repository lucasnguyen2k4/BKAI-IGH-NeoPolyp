import argparse
import os
import torch
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, PILToTensor, InterpolationMode, ToPILImage
import segmentation_models_pytorch as smp
import torch.nn.functional as F
from torch.utils.data import Dataset

class UNetSingleImageDataset(Dataset):
    def __init__(self, image_path, transform):
        super(UNetSingleImageDataset, self).__init__()
        self.image_path = image_path
        self.transform = transform
        
    def __getitem__(self, index):
        data = Image.open(self.image_path)
        h, w = data.size[1], data.size[0]
        data = self.transform(data) / 255
        data = data.unsqueeze(0)
        return data, self.image_path, h, w
    
    def __len__(self):
        return 1
    
def load_eval_model(device):
    model = smp.UnetPlusPlus(
        encoder_name="resnet34",        
        encoder_weights=None,     
        in_channels=3,                  
        classes=3     
    )
    checkpoint = torch.load('unet_model.pth',weights_only=True, map_location=device)
    # Fix "module."
    state_dict = checkpoint["model"]
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(new_state_dict)
    model.to(device)
    return model.eval()

def generate_mask_img(model, transform, image_path):
    single_image_dataset = UNetSingleImageDataset(image_path, transform)
    data, img_path, h, w = single_image_dataset[0]
    with torch.no_grad():
        output = model(data)[0]
    if not os.path.isdir("output"):
        os.mkdir("output")
    mask2img = Resize((h, w), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(output, 0)).permute(2, 0, 1).float()))
    image_id = os.path.basename(image_path).split('.')[0]
    mask2img.save(os.path.join("output", f"mask_{image_id}.png"))

parser = argparse.ArgumentParser(description="Inference script for UNet++ with ResNet encoder.")
parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
                     PILToTensor()])

model = load_eval_model(device)
generate_mask_img(model, transform, args.image_path)