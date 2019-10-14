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



## Related Work
With a drastic boom in the deep learning field, this group of algorithms has considerably improved the state of the art in image classification. Some Convolutional Neural Network Architectures can achieve fairly high accuracy on a complicated image dataset. ResNet, for example, successfully obtained a Top-5 classification accuracy of 96.53% on ImageNet. However, compared with traditional image classification with the ImageNet dataset, this Google Landmark Recognition Challenge has some contrasting differences. The number of classes in the Landmark Change is 20 times bigger than the class size of ImageNet, but the size of training examples per class in the given "Google Landmarks" dataset might be quite small. To address these problems, the previous competitors in this challenge tried different combinations of CNN architectures including ResNet50, Inception-ResNet, and SEResNet. The highest accuracy achieved in this year's challenge is about 0.37 in Kaggle Leader-Board score which again indicates a high difficulty level of this classification task. 

## Methods
### Dataset

- how to acquire dataset
	- The dataset, Google-Landmarks-v1, provided by Google, contains 5 million images of more than 200,000 different landmarks. The images were collected from photographers around the world who labeled their photos and supplemented them with historical and lesser-known images from Wikimedia Commons. 

- Data Cleaning
	- For the data cleaning stage, we are considering adopting similar strategies mentioned by Ozaki et al. The first step is to remove all classes with no more than 3 training samples (53,435 classes in total). Then by applying spatial verification to the filtered images by k nearest neighbor search, we expect the cleaned dataset contains around 2 million images with roughly 100,000 labels. 


### what network architecture to experiment/innovate
- Considering using ResNet-101 and SE-ResNeXt-101 as backbones trained with cosine-softmax / or softmax based losses.



## Expected Results
After applying the data cleaning method, the training data should not contain classes with the size of the training samples smaller than three. Visually unrelated images within the same class should also be discarded. After the meticulous construction of models and tuning of the parameters, we expect our model to achieve around 0.25 in the Kaggle leaderboard score.  

reference

