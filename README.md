# Libtorch Custom Dataset
This is a short example on how to generate custom datasets for libtorch. The `CustomDataset` class in [custom_dataset.h](custom_dataset.h) implements a `torch::data::Dataset`. It loads the image locations from [file_names.csv](file_names.csv) into a `std::vector<std::tuple<std::string, int>>`, so that the `CustomDataset` can load images at runtime with the [get](https://github.com/mhubii/libtorch_custom_dataset/blob/cd3d1028d074bf068924c82387d4520708b7ea8b/custom_dataset.h#L23) method using OpenCV. You may want to change this and load all images to the RAM, since this may significantly slow down the training if you do not use SSDs.

<br>
<figure>
  <p align="center"><img src="data/apples/img2.jpg" width="20%" height="20%" hspace="100"><img src="data/bananas/img0.jpg"      width="20%" height="20%" hspace="100"></p>
  <figcaption>Fig. 2: (Left) An apple in the dataset, (Right) a banana in the dataset.</figcaption>
</figure>
<br><br>

# Build
Make sure to get libtorch running. For a clean installation from Anaconda, checkout this short [tutorial](https://gist.github.com/mhubii/1c1049fb5043b8be262259efac4b89d5), or this [tutorial](https://pytorch.org/cppdocs/installing.html), to only download the binaries.

Clone this repository
```shell
git clone https://github.com/mhubii/libtorch_custom_dataset.git
cd libtorch_custom_dataset
```

Build the executables
```shell
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
make
```
Train the classifier
```shell
cd build
./train
```
Classify an image
```shell
cd build
./classify filename
# for example
./classify ../data/apples/img0.jpg
```

# Notes
The dataset is a modified version of a dataset that can be found on [Kaggle](https://www.kaggle.com/sriramr/apples-bananas-oranges).
Especially the training loop is inspired by an implementation of Peter Goldsburough for the MNIST dataset in the PyTorch [example repository](https://github.com/pytorch/examples/tree/master/cpp/mnist).
