AirNet-SNL Examples
===================

Check out the examples below!

Example 1: Training AirNet-SNL
------------------------------

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

Example 2: Running inference with AirNet-SNL
--------------------------------------------

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