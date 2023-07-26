# Twister_V2
Twister is the internal codename for this paper's topic. Please find the paper in this Github repository.

V2 is the codename for the local development through the use Kaggle.com GPU.

*By Martijn Elands, Luc Bams, Dr. Jerry Spanakis, and Dr. Mirela Popa, Maastricht University (UM) &amp; Mediaan*

## Introduction
Twister is basically about twisting the input space using different techniques to more efficiently train the machine learning model. We can efficiently initialize a machine learning model by keywords or class descriptions (that make up the class embeddings). The goal is to more efficiently train when data is sparse and imbalanced. Research has shown that it works for balanced datasets, so it is possible on a different task/setting. Please see my BSc thesis document for a more detailed description.

## Getting Started
1. It is highly recommended to use Poetry for install and manage all dependencies.
2. Python 3.10.11
3. GPU usage is recommended to speed up the training of transformer models. It is unnecessary to manually specify this as the software should automatically detect if a GPU is available and otherwise use the CPU.

Clone repo to your local environment or to Azure. If Poetry is already installed, then you can use ```poetry install``` to install all dependencies. I have also included a requirements.txt (which might be more useful for Azure).

The dependencies ```cluestar```, ```umap-learn``` and ```keybert``` might be holding back further development. These were used for dataset exploration. Feel free to remove them when installing through Poetry.

### Structure of the code
Since the code was developed on Kaggle with local usage of classes, I have put the classes into the src/ directory. The notebooks (from Kaggle) were modified to work locally. The notebooks will run the class files in the source folder. Thanks to the use of notebooks, it was possible to incoperate visual representation of what is happening and explanations.

### Version overview
This section of the README should give a quick overview of which version of notebook/python file works with what implementation

#### Notebooks
csr-notebook-XX:
- V6: smallBERT
- V7: smallBERT+F
- V8: shotBERT
- V9: shotBERT+F

#### Python files
CustomImplementationXX:
- V1: version with K-way (CE)
- V2: version with K-way (CE) and matching classifier (BCE)
- V3: version with TWO K-way (CE & Focal) and matching classifier (BCE)


## Syncing to Azure DevOps:
Clone repository from Azure DevOps by using the \"Clone --> IDE --> Clone in VS Code or VS Studio Code\"

Then setting appropiate git config:

```git config --local user.name "Your Name"```

```git config --local user.email name@domain.top-level-domain```