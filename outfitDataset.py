from torch.utils.data import Dataset
from PIL import Image

class OutfitDataset(Dataset):

    def __init__(self, pairs, processor):
        self.pairs = pairs
        self.processor = processor

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):

        pair = self.pairs[idx]

        top_img = Image.open(pair["top"]).convert("RGB")
        bottom_img = Image.open(pair["bottom"]).convert("RGB")

        top_inputs = self.processor(images=top_img, return_tensors="pt")
        bottom_inputs = self.processor(images=bottom_img, return_tensors="pt")

        return {
            "top": top_inputs["pixel_values"].squeeze(0),
            "bottom": bottom_inputs["pixel_values"].squeeze(0),
            "label": pair["label"]
        }
    
    