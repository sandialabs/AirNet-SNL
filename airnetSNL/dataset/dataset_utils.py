import numpy as np
from skimage.transform import resize
import torch
from typing import List


def calculateAngles(nAngles: int) -> torch.tensor:
    """Return array of angles in [0, 360) degrees.

    Args:
        nAngles (int): Number of desired angles.

    Returns:
        angles (tensor): Uniformly spaced angles, excluding 360 degrees.

    """
    stepSize = 360 / nAngles
    angles = torch.arange(0, 360, stepSize)
    return angles[:-1]  # remove angle 360 (same as angle 0)


def decimateAngles(nAnglesFull: int, downsample: int) -> torch.tensor:
    """Decimate an array of angles.

    Args:
        * nAnglesFull (int): Number of angles at full resolution.
        * downsample (int): Downsampling factor.

    Returns:
        angles (tensor): Array of downsampled angles.

    Example:
        As an example, suppose the full set contains 451 angles.
        Below are various downsampling factors.
            * 4x : 113 views
            * 8x : 57 views
            * 16x : 29 views
            * 32x : 15 views

    """
    anglesFull = torch.linspace(0, 2 * np.pi, nAnglesFull + 1)
    anglesFull = anglesFull[:-1]
    angles = anglesFull[::downsample]
    return angles


def sampleSinograms(sinograms: torch.tensor,
                    rowRange: List[int]):
    """Calculate train or test set from a subset of rows.

    Args:
        * sinograms (tensor): Sampled row-by-row.
        Dimensions are [batchSize, nAngles, nColumns].
        * rowRange (List[int]): Row range to sample.
        Dimensions are [startRow, endRow].

    Returns:
        Sampled sinogram tensor.

    """
    startRow, endRow = rowRange
    return sinograms[startRow:endRow, :, :]


def decimateSinograms(sinograms: torch.tensor,
                      downsample: int):
    """Decimate the sinograms by angle.

    Args:
        * sinograms (tensor): Dimensions are
        [batchSize, nChannels, nRows (nAngles), nCols].

        * downsample (int): Downsampling factor

    Returns:
        Decimated sinogram tensor.

    """
    decimated = sinograms[:, :, ::downsample, :]
    return decimated


def resizeSinograms(sinograms: np.array, nRows: int = 128):
    '''Reize projection images to (nRows, nRows).

    Args:
        * sinograms (array): Dimensions are
        [nRows, nAngles, nRows (=nColumns)]

        * nRows (int): Desired image size

    Returns:
        Resized sinogram

    '''
    nAngles = sinograms.shape[1]
    resized = np.zeros((nRows, nAngles, nRows))

    for a in range(nAngles):
        resized[:, a, :] = resize(sinograms[:, a, :],
                                  (nRows, nRows),
                                  anti_aliasing=True)
    return resized


def getMask(imgShape: List[int]):
    """Return a mask of an inscribed circle in the image.

    Args:
        imgShape (List[int]): [nRows, nCols]

    Returns:
        Mask of 1 inside circle and 0 outside circle

    """
    n = imgShape[2]
    if n % 2 == 0:
        begin = -n // 2
        end = n // 2 - 1
    else:
        begin = -(n - 1) // 2
        end = (n - 1) // 2

    xAxis = torch.arange(begin, end + 1).type(torch.cuda.FloatTensor).cuda()
    X1, X2 = torch.meshgrid(xAxis, xAxis)
    X1 = X1.float()
    X2 = X2.float()

    distance = torch.sqrt(X1 ** 2 + X2 ** 2)
    distance[distance > end] = -1
    distance[distance != -1] = 1
    distance[distance == -1] = 0

    return distance
