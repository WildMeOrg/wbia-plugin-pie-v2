# -*- coding: utf-8 -*-
from torch.utils.data import Dataset
from skimage import transform as skimage_transform
from skimage import color
import imageio
import numpy as np


class AnimalNameWbiaDataset(Dataset):
    """Dataset to load animal data from COCO format for inference.
    Used in plugin.
    """

    def __init__(
        self,
        image_paths,
        names,
        bboxes,
        viewpoints,
        target_imsize,
        transform,
        fliplr=False,
        fliplr_view=None,
    ):
        self.image_paths = image_paths
        self.bboxes = bboxes
        self.names = names
        self.target_imsize = target_imsize
        self.transform = transform
        self.viewpoints = viewpoints

        if fliplr:
            assert isinstance(fliplr_view, list) and all(
                isinstance(item, str) for item in fliplr_view
            )

        self.fliplr = fliplr
        self.fliplr_view = fliplr_view

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = imageio.imread(self.image_paths[idx])
        if image is None:
            raise ValueError('Fail to read {}'.format(self.image_paths[id]))
        
        # Ensure image is in RGB format
        if image.ndim == 2:
            image = color.gray2rgb(image)
        elif image.shape[2] == 4:
            image = color.rgba2rgb(image)

        # Crop bounding box area
        x1, y1, w, h = self.bboxes[idx]
        image = image[y1 : y1 + h, x1 : x1 + w]
        if min(image.shape) < 1:
            # Use original image
            image = imageio.imread(self.image_paths[idx])
            self.bboxes[idx] = [0, 0, image.shape[1], image.shape[0]]

        # Resize image
        image = skimage_transform.resize(
            image, self.target_imsize, order=3, anti_aliasing=True
        )

        # Flip image if a model have been trained on one specific view
        if self.fliplr:
            if self.viewpoints[idx] in self.fliplr_view:
                image = np.fliplr(image)

        if self.transform is not None:
            image = self.transform(image.copy())
        return image, self.names[idx]
