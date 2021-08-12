import airnetSNL.model.train_loop as tl
# import airnetSNL.model.airnet_snl as snl
import airnetSNL.dataset.dataset_utils as du
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch import optim


# def test_training():
#     angles = du.decimateAngles(nAnglesFull=451,
#                                downsample=8)
#     imgSize = 336
#     batchSize = 10
#     model = snl.AirNetSNL(angles=angles,
#                           n_iterations=12,
#                           n_cnn=10,
#                           imgSize=imgSize,
#                           batchSize=batchSize,
#                           includeSkipConnection=False)

#     optimizer = optim.Adam(model.parameters(), lr=1e-5)

#     trainSinograms = torch.zeros(100, 1, len(angles), imgSize)
#     trainImages = torch.zeros(100, 1, imgSize, imgSize)
#     trainSet = TensorDataset(trainSinograms, trainImages)
#     trainLoader = DataLoader(trainSet, batch_size=batchSize)

#     beforeTraining = {}
#     for i, (name, param) in enumerate(model.named_parameters()):
#         if param.requires_grad:
#             beforeTraining[name] = param.data.clone()

#     tl.train_model(model=model,
#                    optimizer=optimizer,
#                    train_loader=trainLoader,
#                    nEpochs=1,
#                    saveModel=False,
#                    resumeFrom=0,
#                    saveFilePath='./testModel.pth')

#     checkNotEqual = [False] * len(beforeTraining)
#     afterTraining = {}
#     for i, (name, param) in enumerate(model.named_parameters()):
#         if param.requires_grad:
#             afterTraining[name] = param.data

#         checkNotEqual.append(not torch.equal(beforeTraining[name],
#                                              afterTraining[name]))

#     # Check that parameters changed before and after training.
#     assert any(checkNotEqual)


# def test_inference():
#     angles = du.decimateAngles(nAnglesFull=451,
#                                downsample=8)
#     imgSize = 336
#     batchSize = 10
#     totalSamples = 100
#     model = snl.AirNetSNL(angles=angles,
#                           n_iterations=12,
#                           n_cnn=10,
#                           imgSize=imgSize,
#                           batchSize=batchSize,
#                           includeSkipConnection=False)

#     modelFile = './testModel.pth'
#     torch.save(model.state_dict(), modelFile)

#     testSinograms = torch.zeros(totalSamples, 1, len(angles), imgSize)
#     testImages = torch.zeros(totalSamples, 1, imgSize, imgSize)
#     testSet = TensorDataset(testSinograms.cpu(), testImages.cpu())
#     testLoader = DataLoader(testSet, batch_size=batchSize)

#     preds = tl.run_inference(testLoader, model, modelFile, isFileDict=False)
#     assert list(preds.shape) == [totalSamples, 1, imgSize, imgSize]
