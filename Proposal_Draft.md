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
### Data Cleaning
### Model Structures
### Ranking



## Expected Results
After applying the data cleaning method, the training data should not contain classes with size of the training samples smaller than three. Visually unrelated images within the same class should also be discarded. After meticulous construction of models and tuning of the parameters, we expect our model to achieve around 0.25 in Kaggle leaderboard score.  
