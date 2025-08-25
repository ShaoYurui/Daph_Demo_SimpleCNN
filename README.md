# Installation Guide

Follow the steps below to set up the environment and run the demo.


## Step 1: Install Anaconda
Download and install Anaconda from the official website:  
[Anaconda Download](https://www.anaconda.com/download/success)


## Step 2: Install PyTorch
Open the **Anaconda Terminal** and install PyTorch using the command provided on the official website:  
[PyTorch Installation Instructions](https://pytorch.org/)

**[IMPORTANT]** Run the installation command directly in the Anaconda Terminal.


## Step 3: Install Visual Studio Code
Download and install **Visual Studio Code**:  
[VS Code Download](https://code.visualstudio.com/download)

Next, install the following extensions from the **VS Code Marketplace**:
- [Python Extension](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
- [Jupyter Extension](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)


## Step 4: Run the Demo
1. Navigate to: src/Daph sandbox.ipynb

2. In jupyter notebook, select the **kernel** to be Python from your Anaconda environment.
3. Click **Run All** to execute the notebook.


### Expected Output
You should see a classification report similar to the following:

```
Classification Report:
                precision recall  f1-score   support

       0        0.9393    0.9857    0.9620     10453
       1        0.3968    0.1283    0.1939       764

accuracy                            0.9273     11217

macro avg       0.6680    0.5570    0.5779     11217
weighted avg    0.9023    0.9273    0.9096     11217

Validation CrossEntropyLoss: 0.26292428374290466
Macro F1 Score: 0.5779121166953458
```

