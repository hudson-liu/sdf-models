# sdf-models
A small experiment on testing whether a GNO coupled with a Transolver performs better than an FNOGNO on SDF prediction.

## dataset info
Our dataset consists of a mesh along with associated SDFs calculated at various query points. In our dataset, these query points are along a regular grid, though this does not necessarily have to be the case. This regular grid spans a 1.1x-scaled version of the bounding box of the surface mesh, varying between meshes, all of which contain exactly 262144 datapoints. There are a total of 3598 training examples. The directory structure of the dataset is as follows:
```bash
└── /resnick/scratch/hliu9
    ├── mesh-4k
    │   ├── [hash]_processed.obj
    │   ├── [hash]_processed.obj
    │   ├── ...
    │   └── [hash]_processed.obj
    └── sdf-4k-h64
        ├── [hash]_processed.csv
        ├── [hash]_processed.csv
        ├── ...
        └── [hash]_processed.csv
```

**Processing**: We use Open3D for processing meshes; all information other than the surface's point cloud itself is ignored. Each point cloud is individually normalized to [-1, 1]^3, and the corresponding SDF grid for that mesh is normalized to [-1.1, 1.1]^3. A single hold-out fold is used for testing the model; then, the proportion of data specified in "args.split" is used, from the available training pool, to train the model. E.g., in our case, there are 21 folds, and one fold is used for testing, so, if args.split=0.2, then 4 folds out of the 20 training folds will actually be used during training. Each preprocessed fold is saved as a npz file.

**Dataset Loading**: The entire dataset is pre-loaded into memory before training begins. We write our own torch.Dataset obj since the MeshDataModule from NeuralOperators assumes constant query points. Everything is stored in torch.float32. Our dataset is small enough such that data sharding is unnecessary.

## GNO-Transolver
The main GNO-Transolver model is implemented as a torch module. The graph neural operator (GNOBlock from neuralop library) is fed, as input, a 2-tuple (y, x) where y is the surface point cloud. This cannot be batched; as in, the shape of y (and x) should not be "(num_batches, num_points, 3)", but rather "(num_points, 3)". 

## GNO-FNO
The FNOGNO is taken straight from the neuraloperator library.
