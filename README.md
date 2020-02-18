# Transfer Learning with PyTorch

Meet Taz:
![](image/IMG.JPG)

This code is using a well trained net Resnet18 to slove a new classification problem:
Is there Taz in the picture? or no Taz?

# Why transfer learning?

Because the amount of images data is too small to train a good model from the start + not using GPU,
It can be better to use a model that was already trained on lots of data, and use its learned weights that can already detect from simple to complex features.

The main steps in transfer learning:

- Choose the trained model to use - here I use Resnet18 from: [https://pytorch.org/hub/pytorch_vision_resnet/]
- Prepare the data to match as an input to the trained model, with the normalization needed.
  Here the normalization is mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225]
- Decide if you want to fine tuning the model's parameters and train the whole net with the learned parameters as the initial parameters.
  Or the freeze part of the net (usually the start that detects simple features).
  Here I froze the whole net and change the last layer to fit my classification problem, the parameters of this last layer were trained.
