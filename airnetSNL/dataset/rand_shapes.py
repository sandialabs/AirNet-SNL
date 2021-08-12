import numpy as np
from skimage.draw import random_shapes
import torch
from torch.utils.data import Dataset
from torch_radon import Radon


class RandomShapeDataset(Dataset):
    """
    Generate random shapes for training and testing.

    Args:
        * imgSize (int): Number of rows / cols in image
        * maxShapes (int): Number of shapes in image
        * nImg (int): Number of images in dataset
        * angles (torch.tensor): View angles
        * idxOffset (int): Seed for random shape generation
        * scaleFactor (int): Scale of pixel values

    """
    def __init__(self,
                 angles: np.array,
                 imgSize: int = 128,
                 maxShapes: int = 10,
                 nImg: int = 260,
                 idxOffset: int = 0,
                 scaleFactor: int = 1000):

        self.imgSize = imgSize
        self.maxShapes = maxShapes
        self.radon = Radon(imgSize,
                           angles,
                           clip_to_circle=True,
                           det_count=imgSize)
        self.nImg = nImg
        self.angles = angles
        self.idxOffset = idxOffset
        self.scaleFactor = scaleFactor

    def __len__(self):
        return self.nImg

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index number of image to generate

        Returns:
            sinogram (torch.tensor): Array of shape (nPixels, nViews)
            img (torch.tensor): Array of shape (nRows, nCols)

        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if idx >= self.nImg:
            raise ValueError(f'Exceeded {self.nImg} images')

        seed_idx = idx + self.idxOffset
        img, _ = random_shapes((self.imgSize, self.imgSize),
                               max_shapes=self.maxShapes,
                               shape=None,
                               multichannel=False,
                               random_seed=seed_idx,
                               allow_overlap=False)

        img = torch.tensor(img).unsqueeze(0).unsqueeze(0)
        img = img.type(torch.cuda.FloatTensor)

        img = invertAndZero(img, invert=True)

        if torch.max(img) != 0:
            img = img / torch.max(img) * self.scaleFactor

        sinogram = self.radon.forward(img)

        return sinogram.squeeze(0).type(torch.cuda.FloatTensor), \
            img.squeeze(0).type(torch.cuda.FloatTensor)


def invertAndZero(img, invert=True):
    '''
    Calculate (1 - img), assuming image values in [0, 1].
    Zero out the image outside of an inscribed circle.

    '''
    dtype = torch.cuda.FloatTensor
    TENSOR_SCALE_FACTOR = 255
    n = img.shape[2]
    if n % 2 == 0:
        begin = -n // 2
        end = n // 2 - 1
    else:
        begin = -(n - 1) // 2
        end = (n - 1) // 2

    mask = -1 * torch.ones(img.shape).type(dtype)

    x = torch.arange(begin, end + 1).type(dtype)
    X1, X2 = torch.meshgrid(x, x)
    X1 = X1.float()
    X2 = X2.float()
    distance = torch.sqrt(X1 ** 2 + X2 ** 2)

    distance[distance > end] = -1
    nSamples = img.shape[0]

    for ss in range(nSamples):
        mask[ss, 0, :, :] = distance

    if invert:
        zeroed = TENSOR_SCALE_FACTOR - img
    else:
        zeroed = img

    zeroed_tensor = zeroed.masked_fill(mask == -1, 0)

    return zeroed_tensor
