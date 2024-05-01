# Hot Dog Detector

Richa Namballa

## Introduction

Explain your motivation for pursuing this project. You should include background information on the task.

[Motivation](https://www.youtube.com/watch?v=ACmydtFDTGs&ab_channel=HBO)

## Dataset

You do not have to upload the dataset to your GitHub repository, but you should include clear instructions on how to access the data, download it, and set it up in your repository.

## Model

Explain the model architecture that your project uses. Be sure to include an architecture diagram here!

## Using this Repository

Please provide detailed instructions on how your codebase works. Ideally, someone should be able to recreate your model by following this guide.

### Requirements

Give instructions on setting up the environment.

```
conda env create -f environment.yml
conda activate hotdog
```

### Training the Model

```python
model = u.build_model()

# Adjust class weights based on the dataset balance
class_weights = {0: 1, # o
                 1: 1} # n

history, model = u.train_model(model, x_train, y_train, x_val, y_val, 
                      epochs=30, batch_size=64, class_weights=class_weights)

u.plot_loss(history)
```


### Using the Model

```python
# Load demo data from the folder
file_paths = ["../Data/demo.json"]
x_data, y_data = u.load_data(file_paths) 
```

```python
# Load model from the folder
from tensorflow.keras.models import load_model
model = load_model('Handpose-Recognition.h5')
```

```python
import numpy as np

# Classify handpose
# sample random handpose from the demo dataset
idx = np.random.randint(0, len(y_data), size=1)[0] 
handpose_features = x_data[idx][np.newaxis, :] 
print('This handpose is not the target handpose.' if y_data[idx]==0 else 
      'This handpose is the target handpose.')

# Use your model to make predictions
y_pred = u.classify_handpose(model, handpose_features)

if y_pred[0] == 1:
    print('The model says this handpose is the target handpose.')
else:
    print('The model says this handpose is not the target handpose.')
```

<img width="400" alt="截屏2024-04-24 18 54 23" src="https://github.com/RubyQianru/final-project-example/assets/142470034/4830b139-ae45-4b60-84c1-669fc4667675">

## References

1. Transfer learning model is originally based on [TensorFlow Hand Pose Detection Model](https://github.com/tensorflow/tfjs-models/tree/master/hand-pose-detection)
2. Model report and testing code is based on [DL4M Homework 1](https://github.com/dl4m/homework-1-RubyQianru)

