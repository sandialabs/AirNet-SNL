from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import torch


def calculateMSE(gt: torch.tensor, img: torch.tensor) -> float:
    """Calculate mean squared error.

    Args:
        * gt (tensor): Ground truth tensor.
        * img (tensor): Prediction. Same size as gt.

    Returns:
        Mean squared error (float).

    """
    return mse(gt.cpu().numpy(), img.cpu().numpy())


def calculateSSIM(gt: torch.tensor, img: torch.tensor) -> float:
    """Calculate structural similarity image metric.

    Args:
        * gt (tensor): Ground truth tensor.
        * img (tensor): Prediction. Same size as gt.

    Returns:
        Structural similarity image metric (float).

    """
    return ssim(gt.cpu().numpy(), img.cpu().numpy(),
                data_range=img.cpu().numpy().max() - img.cpu().numpy().min())


def calculatePSNR(gt: torch.tensor, img: torch.tensor) -> float:
    """Calculate peak signal to noise ratio.

    Args:
        * gt (tensor): Ground truth tensor.
        * img (tensor): Prediction. Same size as gt.

    Returns:
        Peak signal to noise ratio (float).

    """
    return psnr(gt.cpu().numpy(), img.cpu().numpy())
