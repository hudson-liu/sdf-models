# sdf-models
A small experiment on testing whether a GNO coupled with a Transolver performs better than an FNOGNO on SDF prediction.

## dataset info
Our dataset consists of a mesh along with associated SDFs calculated at various query points. In our dataset, these query points are along a regular grid, though this does not necessarily have to be the case. This regular grid spans a 1.1x-scaled version of the bounding box of the surface mesh, varying between meshes, all of which contain exactly 262144 datapoints. There are a total of 3598 training examples. The directory structure is as follows:

The dataset is split into 10 folds during preprocessing, of which any can be used for validation.

**Processing**: We use Open3D for processing meshes; all information other than the surface's point cloud itself is ignored. Each point cloud is individually normalized to [-1, 1]^3, and the corresponding SDF grid for that mesh is normalized to [-1.1, 1.1]^3. After processing, each fold is saved as a series of npy files.

**Dataset Loading**: The entire dataset is pre-loaded before training begins. We write our own torch.Dataset obj since the MeshDataModule from NeuralOperators assumes cnnstant query points. Everything is stored on RAM in torch.float32. Our dataset is small enough such that data sharding is unnecessary. GNO takes in y

# for next step, i can just look at what they did for carcfd and basically copy that
# i think what strategically makes the most sense is to first do fnogno

## GNO-Transolver
The main GNO-Transolver model is implemented as a torch module, with inputs of shape 
The graph neural operator (GNOBlock from neuralop library) is fed, as input, a 2-tuple (y, x) where y is the surface point cloud

# GNO-FNOl
conversely, if we couple 
