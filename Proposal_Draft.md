# Proposal Draft
## Instruction 
- Your project proposal must answer 
  - what you want to do
  - why you want to do it
  - what others have done
  - how you will do it. For how, outline a technical implementation plan 
    - how to acquire dataset, what network architecture to experiment/innovate
  - what results you expect

## Outline
- Title and abstract
- Introduction
- Related Work
- Technical overview (methods)
- Expected results
- Reference



## Introduction 

While ImageNet attracts a lot of attention and a lot of models achieve high accuracy, the computer vision area lacks models for recognizing landmarks. We will be building a convolutional network model using Google-landmarks dataset since it is the largest landmark dataset available. This posts a huge challenge due to the smaller size of the dataset compare to Imagenet, and also the shared features between landmarks that were built in the same era and architectural style. 



## Related Work

There have been numerous innovations in the architectures of convolutional neural networks that drastically boosted the accuracy of some complicated image classification tasks. ResNet, for example, successfully obtained a Top-5 classification accuracy of 96.53% on ImageNet. Later innovations including VGG, Inception and different variations of ResNet have also improved the accuracy and efficiency in the image classification field. By using the combinations of these architectures, the previous competitors in the Kaggle Google Landmark Recognition Challenge were able to achieve fairly good results. The solution given by the first-place group used ResNet-101, ResNeXt-101 (64×4d), SE-ResNet-10, SE-ResNeXt-101 (32×4d) and SENet-154 as their backbone networks. The third-place solution used FishNet-150 [20], ResNet-101 and SE-ResNeXt-101 as backbones. The evaluation metrics for the final test is the Global Average Precision (GAP). Both groups achieved around 0.3 GAP score. 

## Methods
### Dataset

- how to acquire dataset
	- The dataset, Google-Landmarks-v1, provided by Google, contains 5 million images of more than 200,000 different landmarks. The images were collected from photographers around the world who labeled their photos and supplemented them with historical and lesser-known images from Wikimedia Commons. 

- Data Cleaning
	- For the data cleaning stage, we are considering adopting similar strategies mentioned by Ozaki et al. The first step is to remove all classes with no more than 3 training samples (53,435 classes in total). Then by applying spatial verification to the filtered images by k nearest neighbor search, we expect the cleaned dataset contains around 2 million images with roughly 100,000 labels. 


### what network architecture to experiment/innovate
- Considering using ResNet-101 and SE-ResNeXt-101 as backbones trained with cosine-softmax / or softmax based losses.



## Expected Results
After applying the data cleaning method, the training data should not contain classes with the size of the training samples smaller than three. Visually unrelated images within the same class should also be discarded. After the meticulous construction of models and tuning of the parameters, we expect our model to achieve around 0.25 in GAP score.  

reference

