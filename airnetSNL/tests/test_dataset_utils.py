import airnetSNL.dataset.dataset_utils as du
import torch


def test_decimate_angles():
    nAnglesFull = 451
    downsample = 4
    angles = du.decimateAngles(nAnglesFull, downsample)
    assert len(angles) == 113


def test_sample_sinograms():
    sinograms = torch.zeros((400, 57, 336))
    sample = du.sampleSinograms(sinograms, [0, 200])
    assert sample.shape[0] == 200
