import sys
sys.path.append('..')

from airnetSNL.dataset.dataset_utils import sampleSinograms
from glob import glob
from imageio import imread
import numpy as np
import os
import sys
import torch
from tqdm import tqdm
from typing import List


def saveMultipleExperiments(projDirs: List[str],
                            trainSinoFile: str,
                            testSinoFile: str):
    """Load projections, and save them in train and test sets.
    Uses the top half of each projection for training and
    the bottom half for testing.

    Args:
        * projDirs (List[str]): Directories with tif projections.
        * trainSinoFile (str): Filename to save train projections.
        * testSinoFile (str): Filename to save test projections.

    Returns:
        Saves projections in two files for training and testing.

    """
    for p, projDir in enumerate(projDirs):
        sinograms = loadSinograms(projDir)
        nRows = sinograms.shape[0]
        trainBatch = sampleSinograms(sinograms,
                                     [0, nRows // 2])
        testBatch = sampleSinograms(sinograms,
                                    [nRows // 2, nRows])
        if p == 0:
            trainSinograms = trainBatch
            testSinograms = testBatch
        else:
            trainSinograms = torch.cat((trainSinograms,
                                        trainBatch), dim=0)
            testSinograms = torch.cat((testSinograms,
                                       testBatch), dim=0)

    torch.save(trainSinograms.cpu(),
               trainSinoFile)
    torch.save(testSinograms.cpu(),
               testSinoFile)


def loadSinograms(projDir: str) -> torch.tensor:
    """Return array of 2D sinograms from row-by-row projections

    Args:
        projDir: Directory with .tif files

    Returns:
        * sinograms (tensor):
        (batchSize = nRows) x (nChannels = 1) x nAngles x nColumns

    Notes:
        For absorption and dark field,
        take the neg-log of each projection.
        For differential or integrated phase, negate each projection
        to make most values positive.

    """
    tifs = sorted(glob(os.path.join(projDir, '*.tif')))
    tifs = tifs[:-1]  # Exclude last angle (360 degrees).
    nAngles = len(tifs)

    projection = imread(tifs[0])

    nRows = projection.shape[0]
    nCols = projection.shape[1]

    sinograms = torch.zeros((nRows, 1, nAngles, nCols))

    with tqdm(total=nRows, file=sys.stdout) as pbar:
        for r in range(nRows):
            pbar.set_description('processed: %d' % (1 + r))
            pbar.update(1)
            for a in range(nAngles):
                projection = imread(tifs[a])

                if np.abs(np.max(projection)) > np.abs(np.min(projection)):
                    # Absorption, dark field
                    projection = -np.log(projection)
                else:
                    # Differential/integrated phase case
                    projection = -projection

                sinograms[r, 0, a, :] = torch.tensor(projection[r, :])

    return sinograms
