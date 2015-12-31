Deep learning for music information retrieval
-----------------------------

This library is work in progress, use it at your own risk. I'm open to feedback!

It is build using python on Lasagne-Theano for deep learning and Essentia for feature extraction.
Currently, MIRdl is for easily doing music classification using any deep learning architecture available on Lasagne or Theano.

**Installation**
 
 Requires having Lasagne-Theano (http://lasagne.readthedocs.org/en/latest/user/installation.html) and Essentia (http://essentia.upf.edu/documentation/installing.html) installed.
 
**Important folders**
- *./data/datasets*: the library expects to have the dataset divided by folders that represent the tag to be predicted. 
- *./data/preloaded*: this directory contains the pickle files storing the datasets in a format readable for the library. The name of the pickle file contains all the parameters used for computing it.
- *./data/results*: this directory stores the following files: **.result** (with training and test results), **.training** (having the training evolution, readable with utils.py!), **.param** (storing all the deep learning parameters used for each concrete experiment) and the **.npz** (where the best trained deep learning model is stored).
 
**Important scripts**
- *runMIRdl.py*: where the network architecture is selected, you can also set the input and training parameters.
- *buildArchitecture.py*: where the Lasagne-Theano network architecture is set.
- *load_datasets.py*: where audios are loaded, formatted and normalized to be fed into the net. 
- *MIRdl.py*: main part of the library where the training happens.
- *utils.py*: it allows visualizing the training results (*./data/results*).

**Steps for using MIRdl**
- **0.0)** Install.

- **0.1)** Understand this tutorial: http://lasagne.readthedocs.org/en/latest/user/tutorial.html. This library is based on it!

- **1)** Download a dataset. Copy it in *./data/datasets*. The library expects to have the dataset divided by folders that represent the tag to be predicted. 
For example, for the GTZAN dataset (http://marsyasweb.appspot.com/download/data_sets/) the library expects:

>./data/datasets/GTZAN/blues
>
>./data/datasets/GTZAN/classical
>
> (...)
>
>./data/datasets/GTZAN/rock,

where each folder contains all the songs for training, testing and validating for each class.

- **2)** Adapt the *load_datasets.py* function to work using your dataset. We recommend you to use first the GTZAN dataset (already implemented) to understand how it works.

- **3)** Set the *runMIRdl.py* parameters and the deep learning architecture in *buildArchitecture.py*.

- **4)** Run *runMIRdl.py*.

- **5)** *[Optional]* Visualize what the net has learned with *utils.py*.  

**Future features**
- Autoencoders support.
- AcousticBrainz.org support.



