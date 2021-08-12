import errno
import numpy as np
import os
import torch
from torch.nn import functional as F


def train_model(model: torch.nn.Module,
                optimizer: torch.optim.Adam,
                train_loader: torch.utils.data.DataLoader,
                nEpochs: int,
                saveModel: bool = False,
                resumeFrom: int = 0,
                saveFilePath: str = '',
                loadFilePath: str = ''):
    """Train a model.

    Args:
        * model (Module): Model to be trained.
        * optimizer (Adam): Optimization parameters, e.g. learning rate.
        * train_loader (DataLoader): Dataset parameters, e.g. batch size.
        * nEpochs (int): Number of epochs for training.
        * saveModel (bool): Save model if loss improves.
        * resumeFrom (int): Resume training from epoch number.
        * saveFilePath (str): Where to save the model.
        * loadFilePath (str): Which model to load.

    Example:
        .. code-block:: python
           :linenos:

           import airnetSNL.model.train_loop as tl
           import airnetSNL.model.airnet_snl as snl
           import airnetSNL.dataset.dataset_utils as du
           import torch
           from torch.utils.data import TensorDataset, DataLoader
           from torch import optim

           angles = du.decimateAngles(nAnglesFull=451,
                                      downsample=8)
           imgSize = 336
           batchSize = 10
           model = snl.AirNetSNL(angles=angles,
                                 n_iterations=12,
                                 n_cnn=10,
                                 imgSize=imgSize,
                                 batchSize=batchSize,
                                 includeSkipConnection=False)

           optimizer = optim.Adam(model.parameters(), lr=1e-5)

           trainSinograms = torch.zeros(100, 1, len(angles), imgSize)
           trainImages = torch.zeros(100, 1, imgSize, imgSize)
           trainSet = TensorDataset(trainSinograms, trainImages)
           trainLoader = DataLoader(trainSet, batch_size=batchSize)

           tl.train_model(model=model,
                          optimizer=optimizer,
                          train_loader=trainLoader,
                          nEpochs=1,
                          saveModel=False,
                          resumeFrom=0,
                          saveFilePath='./testModel.pth')

    """

    def train_one_batch(x_sinogram, y_img_gt):
        y_img_pred = model(x_sinogram)
        loss = F.mse_loss(y_img_pred, y_img_gt)

        optimizer.zero_grad()  # Clears Gradient
        loss.backward()  # Calculate Gradients
        optimizer.step()  # Update Weights

        loss = loss.item()
        return loss

    def train_epoch():
        total_loss = 0

        for x_sinogram, y_img in train_loader:
            x_sinogram = x_sinogram.to('cuda')
            y_img = y_img.to('cuda')
            batch_loss = train_one_batch(x_sinogram, y_img)
            total_loss += batch_loss

        return total_loss

    def save_model(epoch_num):
        state = {
            'epoch': epoch_num,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_per_epoch': loss_per_epoch,
        }
        torch.save(state, saveFilePath)

    model.cuda()
    if not saveFilePath:
        raise ValueError("Need to specify saveFilePath to save model.")

    saved_epoch = 0
    loss_per_epoch = -1 * torch.ones(nEpochs)
    loss_per_epoch = loss_per_epoch.type(torch.cuda.FloatTensor).cuda()

    if resumeFrom > 0:
        if not os.path.isfile(loadFilePath):
            raise FileNotFoundError(errno.ENOENT,
                                    os.stderror(errno.ENOENT),
                                    loadFilePath)
        checkpoint = torch.load(loadFilePath)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        saved_epoch = checkpoint['epoch']
        saved_loss = checkpoint['loss_per_epoch']

        # Initialize loss_per_epoch with saved_loss
        if len(saved_loss) < nEpochs:
            loss_per_epoch = -1 * np.ones(nEpochs)
            loss_per_epoch[:len(saved_loss)] = saved_loss
        print('Loaded model!')

    print(f'saved_epoch = {saved_epoch}, nEpochs = {nEpochs}')

    save_interval = 1
    minLoss = torch.tensor(1e9).type(torch.cuda.FloatTensor).cuda()
    for e in range(saved_epoch, nEpochs):
        total_loss = train_epoch()
        loss_per_epoch[e] = total_loss

        # Save model if loss decreases
        if e % save_interval == 0 and saveModel and total_loss < minLoss:
            minLoss = total_loss
            save_model(e)
            print(f'epoch = {e}, loss = {total_loss}, saved!')
        else:
            print(f'epoch = {e}, loss = {total_loss}')

    # Save if checkpoint before nEpochs
    if saved_epoch < nEpochs and saveModel:
        epoch = nEpochs - 1
        save_model(epoch)
        print('Saved model!')


def run_inference(testLoader, model, filepath, isFileDict=True):
    """Run the AirNet-SNL model by passing it in the argument.

    Args:
        * testLoader (DataLoader): Dataset for running inference.
        * model (nn.Module): Use this model for inference.
        * filepath (str): Saved model weights.
        * isFileDict (bool): True if filepath is a dictionary

    Returns:
        Predictions: Tensor of size [nSamples, 1, imgSize, imgSize].

    Example:
        .. code-block:: python
            :linenos:

            import airnetSNL.model.airnet_snl as snl
            import airnetSNL.dataset.dataset_utils as du
            import torch
            from torch.utils.data import TensorDataset, DataLoader

            angles = du.decimateAngles(nAnglesFull=451,
                                       downsample=8)
            imgSize = 336
            batchSize = 10
            totalSamples = 100

            model = snl.AirNetSNL(angles=angles,
                                  n_iterations=12,
                                  n_cnn=10,
                                  imgSize=imgSize,
                                  batchSize=batchSize,
                                  includeSkipConnection=False)
            model = model.cuda()
            filepath = './model.pth'
            testSinograms = torch.zeros(totalSamples, 1, len(angles), imgSize)
            testImages = torch.zeros(totalSamples, 1, imgSize, imgSize)
            testSet = TensorDataset(testSinograms.cpu(), testImages.cpu())
            testLoader = DataLoader(testSet, batch_size=batchSize)
            y_img_pred = run_inference(testLoader, model, filepath)

    """

    if not filepath:
        raise ValueError("Need to specify filepath to load model.")
    if isFileDict:
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(torch.load(filepath))

    model.eval().cuda()
    totalLoss = 0
    for i, (xSino, yImg) in enumerate(testLoader):
        xSino_gpu = xSino.to('cuda')

        pred = model(xSino_gpu).data.cpu()
        loss = F.mse_loss(pred, yImg)
        loss += loss.item()
        totalLoss += loss
        print(f'totalLoss = {totalLoss}')

        if i == 0:
            preds = pred
        else:
            preds = torch.cat((preds, pred), dim=0)

    return preds
