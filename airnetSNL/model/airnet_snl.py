import numpy as np
import torch
from torch.nn import functional as F
from torch import nn
from torch_radon import Radon
import sys
sys.path.append('..')
from airnetSNL.dataset.dataset_utils import getMask


dtype = torch.cuda.FloatTensor
device = torch.device("cuda")
torch.set_default_tensor_type(dtype)


class CNN(nn.Module):
    """Regularization for sparse-data CT and XPCI CT.

    * The CNN has 3 layers:
    inChannels -> Layer 1 -> n_cnn -> Layer 2 ->
    n_cnn -> Layer_3 -> 1 channel

    Args:
        n_cnn (int): Number of output channels in the 1st and 2nd layers.
        imgSize (int): Number of rows/columns in the input image.
        inChannels (int): Number of input channels to the CNN.

    """
    def __init__(self, n_cnn: int,
                 imgSize: int,
                 inChannels: int):
        super().__init__()
        self.n_cnn = n_cnn
        self.imgSize = imgSize
        self.inChannels = inChannels
        # OutputSize = (N - F)/stride + 1 + pdg*2/stride
        # pdg = (N - (N - F) / stride - 1) * stride / 2
        stride = 1
        kernelSize = 3
        pad = (imgSize - (imgSize - kernelSize) / stride - 1) * stride // 2
        pad = int(pad)
        self.conv1 = nn.Conv2d(in_channels=self.inChannels,
                               out_channels=self.n_cnn,
                               kernel_size=kernelSize,
                               padding=pad).to(device)
        self.conv2 = nn.Conv2d(in_channels=self.n_cnn,
                               out_channels=self.n_cnn,
                               kernel_size=kernelSize,
                               padding=pad).to(device)
        self.conv3 = nn.Conv2d(in_channels=self.n_cnn,
                               out_channels=1,
                               kernel_size=kernelSize,
                               padding=pad).to(device)

    def forward(self, x_concat):
        x = F.relu(self.conv1(x_concat))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class IterationBlock(nn.Module):
    """Simulates one iteration of forward model and CNN regularization.

    Args:
        * n_cnn (int): Number of channels.
        * angles (array): Array of measured angles.
        * imgSize (int): Size of the 2D slice to be reconstructed.
        * batchSize (int): Batch size.
        * inChannels (int): Number of input channels to the CNN.
        * includeSkipConnection (bool): Bypass the CNN block.

    """
    def __init__(self, n_cnn: int,
                 angles: np.array,
                 imgSize: int,
                 batchSize: int,
                 inChannels: int,
                 includeSkipConnection: bool = False):
        super(IterationBlock, self).__init__()
        self.n_cnn = n_cnn
        self.step = torch.tensor(1., requires_grad=True).to('cuda')

        self.angles = angles
        self.imgSize = imgSize
        self.inChannels = inChannels
        self.radon = Radon(imgSize,
                           angles,
                           clip_to_circle=True,
                           det_count=imgSize)
        self.batchSize = batchSize
        self.includeSkipConnection = includeSkipConnection
        self.cnn = CNN(self.n_cnn, self.imgSize, self.inChannels).cuda()

    def forward(self, x_sinogram, y_img_prev, y_img_concat_prev):
        Ay_projection = self.radon.forward(y_img_prev)
        difference_projection = Ay_projection - x_sinogram

        filtered_sinogram = self.radon.filter_sinogram(difference_projection)
        fbp = self.radon.backprojection(filtered_sinogram)
        update_img = self.step * fbp

        y_img_update = y_img_prev + update_img

        y_img_concat = torch.cat((y_img_update, y_img_concat_prev), 1)

        prediction_img = self.cnn(y_img_concat)

        y_img = prediction_img
        # Uncomment for skip connections
        # y_img = y_img_update + prediction_img
        return y_img, y_img_update


class AirNetSNL(nn.Module):
    """Computes forward model and regularizes with a CNN for N iterations.

    Args:
        * angles (array): Array of measured angles.
        * n_iterations (int): Number of times to run forward model + CNN.
        * n_cnn (int): Number of output channels for the CNN layers.
        * imgSize (int): Size of the 2D slice to be reconstructed.
        * batchSize (int): Batch size
        * includeSkipConnection (bool): Bypass the CNN block.

    """
    def __init__(self,
                 angles: np.array,
                 n_iterations: int = 12,
                 n_cnn: int = 10,
                 imgSize: int = 128,
                 batchSize: int = 128,
                 includeSkipConnection: bool = False):
        """
        imgInit: 'zeros' or 'fbp'
        """
        super().__init__()
        self.n_iterations = n_iterations
        self.n_cnn = n_cnn
        self.angles = angles
        self.imgSize = imgSize
        self.blocks = nn.ModuleList()
        self.includeSkipConnection = includeSkipConnection
        self.zeroMask = getMask((batchSize, 1, imgSize, imgSize))
        self.batchSize = batchSize

        self.radon = Radon(imgSize,
                           angles,
                           clip_to_circle=True,
                           det_count=imgSize)

        for ii in range(self.n_iterations):
            self.blocks.append(
                IterationBlock(self.n_cnn,
                               self.angles,
                               self.imgSize,
                               self.batchSize,
                               ii + 1,
                               self.includeSkipConnection).cuda())

    def forward(self, x_sinogram_in):
        """
        x_sinogram_in: (nSamples, nChannels, nRows, nCols)
        """
        x_sinogram = x_sinogram_in

        filtered_sinogram = self.radon.filter_sinogram(x_sinogram)
        y0_img = self.radon.backprojection(filtered_sinogram)

        y_img_concat = torch.empty(0).type(torch.cuda.FloatTensor).cuda()
        y_img_prev = y0_img
        for ii in range(self.n_iterations):
            y_img_block, y_img_update = self.blocks[ii](x_sinogram,
                                                        y_img_prev,
                                                        y_img_concat)
            y_img_concat = torch.cat((y_img_update, y_img_concat), 1)
            y_img_prev = y_img_block

        y_img_zeroed = self.zeroMask * y_img_block
        return y_img_zeroed
