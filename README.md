# Libtorch Custrom Dataset
This is a short example on how to generate custom datasets for libtorch. 
<br>
<figure>
  <p align="center"><img src="img/apples/img0.jpg" width="20%" height="20%" hspace="40"><img src="img/bananas/img0.jpg"      width="20%" height="20%" hspace="40"></p>
  <figcaption>Fig. 2: (Left) An apple in the dataset, (Right) a banana in the dataset.</figcaption>
</figure>
<br><br>

# Build
Make sure to get libtorch running.

Clone this repository
```shell
git clone https://github.com/mhubii/libtorch_custom_dataset.git
cd libtorch_custrom_dataset
```
Unzip the data
```shell
tar -zxvf data.tar.gz
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
Especially the training loop is enspired by an implementation of Peter Goldsburough for the MNIST dataset in the PyTorch [example repository](https://github.com/pytorch/examples/tree/master/cpp/mnist).
