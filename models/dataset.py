import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class BirdSoundsSegmentationDataset(Dataset):
    """
    Dataset for the DVAD paper pipeline.
    Loads spectrogram images and their corresponding labeled binary masks
    from the BirdSoundsDenoising dataset.
    
    Dataset structure (from Zenodo):
        Train/ or Valid/
            ├── Images/           # Spectrogram images (STFT magnitude, saved as PNG)
            ├── Masks/            # Binary mask labels (0=noise, 1=clean signal)
            ├── Raw_audios/       # Original noisy audio files
            └── Denoised_audios/  # Clean audio files
    
    For training the segmentation model, we only need Images and Masks.
    """
    def __init__(self, images_dir, masks_dir, img_size=(512, 512), augment=False):
        """
        Args:
            images_dir: Path to the 'Images' folder
            masks_dir: Path to the 'Masks' folder
            img_size: Target image size (paper uses 512x512 for DeepLabV3)
            augment: Whether to apply data augmentation
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.augment = augment

        # Get sorted list of image filenames
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))
        ])
        
        # Verify corresponding masks exist
        valid_files = []
        for f in self.image_files:
            # Try matching with same filename or common mask naming conventions
            mask_candidates = [
                f,                           # Same filename
                f.replace('.png', '_mask.png'),
                f.replace('.jpg', '_mask.png'),
            ]
            found = False
            for candidate in mask_candidates:
                if os.path.exists(os.path.join(masks_dir, candidate)):
                    valid_files.append((f, candidate))
                    found = True
                    break
            if not found:
                # Try to find a mask with the same base name
                base_name = os.path.splitext(f)[0]
                for mask_f in os.listdir(masks_dir):
                    if os.path.splitext(mask_f)[0] == base_name:
                        valid_files.append((f, mask_f))
                        found = True
                        break
                if not found:
                    print(f"Warning: No mask found for image '{f}', skipping.")
        
        self.file_pairs = valid_files
        print(f"Found {len(self.file_pairs)} valid image-mask pairs.")

        # Image transform: resize to img_size, convert to 3-channel RGB, normalize
        # Paper uses [512 × 512 × 3] input for DeepLabV3
        self.image_transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        # Mask transform: resize to img_size, convert to binary tensor
        self.mask_transform = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_name, mask_name = self.file_pairs[idx]
        
        # Load image as RGB (3 channels, as required by paper)
        image = Image.open(os.path.join(self.images_dir, img_name)).convert('RGB')
        
        # Load mask as grayscale
        mask = Image.open(os.path.join(self.masks_dir, mask_name)).convert('L')
        
        # Apply transforms
        image = self.image_transform(image)
        mask = self.mask_transform(mask)
        
        # Convert mask to binary (0 or 1) and squeeze channel dim
        # Threshold at 0.5 to handle any grayscale artifacts
        mask = (mask.squeeze(0) > 0.5).long()  # (H, W) with values {0, 1}
        
        return image, mask
