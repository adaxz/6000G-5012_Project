# Google Landmark Recognition Challenge

- Kaggle Competition Page: [Google Landmark Recognition 2019](https://www.kaggle.com/c/landmark-recognition-2019)
- Applied VGG-16 and ResNet-50 with data augmentation
- Achieved a classification accuracy of 93.77% and 95.34%, respectively.

## Preprocessing

- `preprocessing.ipynb`
    - Select and sampling data
    - Output two files:
        - `train_200.csv`: To increase the model stabil- ity and accelerate the training process, only the images with its corresponding landmark occurrence more than 200 are selected for modeling.
            - The total numbers of images are 630277, 88887, 88887 corresponding to train- ing, validating and testing set. The total number of training labels (i.e. the classes that need to be recognized) is 1066.
        - `train_sample_temp.csv`: used for testing and debugging. Only selected classes containing 1000 images and then randomly sampled images from these classes.
- `seperate_files`:
    - Applied stratified shuffle split to seperate data into
        - training (78% for model training): 630277 images
        - validation (11% for hype-parameters tuning): 88887 images
        - testing (11%, for final evaluation): 88887 images
    - based on the metadata of images in `train_200.csv`, split 'train directory' into training, validation and testing directories using python shutil library
    - Final Directory Tree

# Modeling

- `modeling_stage1.ipynb`:
    - started with a relatively small dataset and fed the dataset into LeNet-5, a very basic and light- weighted ConNet.
    - Images are randomly sampled from each class and loaded batches of those images during train- ing by a custom data generator (detailed python script in `generator.py`)
    - The purpose for this stage is to assure that:
        - all the methods in the pipeline were used correctly.
        - size of images (128x128x3) corresponds with the size set in the input layer
        - the loading process is correct
- `modeling_stage2.ipynb`: Training without Data Augmentation
    - we created a imageDataGenerator object without setting or passing any parameters, which means we were going to use original images for training
    - We built up using two pre-trained models in this stage, VGG-16 and ResNet-50. Since the input size of our data is different from the data in ImageNet, we did not include the top layers in VGG or ResNet.
    - VGG-16
        - we added Global Average Pooling Layer and a fully connected layer with softmax as the activation function at the end
        - Hyper-parameters
            - number of epochs: 8
            - batch size: 128
            - optimizer: Adam with learning rate = 0.0001
            - Top two layers frozen
    - ResNet-50
        - only the final fully connected layer with softmax is added as the activation function in the final layer.
        - Hyper-parameters
            - number of epochs: 8
            - batch size: 128
            - optimizer: Adam
            - learning rate
                - initialized at 0.0001
                - halved at 5th epoch
                - halved again at 7th epoch
            - All layers trainable
- `data_augmentation.ipynb`
    - All the parameters of these transformations were passed into ImageDataGenerator object. With flow_ from_directory, ImageDataGenerator loaded the data from specific directory and perform real-time data augmentation during training.
        - rotation
        - width shift
        - height shift
        - zoom
    - Only use ResNet-50
    - Hyper-parameters:
        - batch size: 128
        - optimizer: adam
        - learning rate: initialized as 1e-4 with step decay
        - all layers trainable

# Environment setup

- Google Cloud Platform
- 8-CPUs, 30 GB virtual machine
- NVIDIA Tesla 100P GPU.
- The average training time for all the models in stage 2 and 3 is around 3.5 hours for total 8 epochs.