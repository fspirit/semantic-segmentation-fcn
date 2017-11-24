# Semantic Segmentation Project

### Setup
##### Frameworks and Packages
Make sure you have the following is installed:
 - [Python 3](https://www.python.org/)
 - [TensorFlow](https://www.tensorflow.org/)
 - [NumPy](http://www.numpy.org/)
 - [SciPy](https://www.scipy.org/)
 - *Imgaug* `pip install imgaug`
 - *OpenCV* `conda install -c conda-forge opencv`
 
I used g2x2large VM on AWS with 'udacity advanced deep learning' AMI.
After launching a VM I used [setup script](setup.sh) to set it up properly.

### Dataset
Download the [Kitti Road dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php) from [here](http://www.cvlibs.net/download.php?file=data_road.zip).  Extract the dataset in the `data` folder.  This will create the folder `data_road` with all the training a test images.

### How to run

Run the following command to run the project:
```
python main.py
```

### Differences with a starter code

I moved all model construction, train and test code to a separate class `FCNModel`.
Also I've abandoned all the tests.

### Augmentation

I decided to use image augmentation to give model more data to train on. I used `imgaug` library
and flip plus random crop augmentations.

Augmentation pipeline:

```
seq = iaa.Sequential([
    iaa.Fliplr(0.5),
    iaa.Crop(percent=(0, 0.1)),
], random_order=True)
```

### Validation set

As it was not entirely clear, whether augmentation helped or not, I decided to extract validation set
from training set and use it to evaluate model performances with/without augmentation and 
different number of epochs.

Here are the results:

Number of epochs | Loss on validation set with augmentation | Loss on validation set without augmentation
--- | --- | ---
10 | 0,24 | 0,15
20 | 0,13 | 0,119
40 | 0,076 | 0,14
80 | 0,087 | --

So when the number of epochs increases, augmentation starts to make difference, as model sees more diverse 
data than a model without augmentation.

### Image results

Some samples:

![1](https://yadi.sk/i/5hvW_vtL3Pzopv)
![2](https://yadi.sk/i/w--8vwzV3PzpLq)
![3](https://yadi.sk/i/YyLUSVaB3PzpN7)
![4](https://yadi.sk/i/mOk6WS-e3PzpQf)

Some samples

### Video

I decided to generate sample video. To do that I did the following:
1. Trained and saved model on server
2. Downloaded saved model to my local computer
3. Run `add_color_to_video.py` script to paint up the video

[Result video](https://yadi.sk/i/qiZg9wyo3PzouZ)

