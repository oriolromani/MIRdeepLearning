import theano, lasagne, csv
import theano.tensor as T
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import buildArchitecture as buildArch

def visualizeWcnn1(name):
    'Visualize weights of the convolutional layer of cnn1'

    ##!!## biases not shown !
    ##!!## deterministic W ?

    # load parameters
    with open(name+'.param', 'rb') as paramFile:
        params = csv.reader(paramFile, delimiter='-')
        count=0;
        for param in params:
            if count==0:
                tmp1=param
                count=count+1
            else:
                tmp2=param
    parameters = {}
    for i in range(len(tmp2)):
        parameters[tmp1[i]] = tmp2[i]

    print("Building network..")
    input_var = T.tensor4('inputs')
    network,netLayers,parameters=buildArch.buildNet(input_var,parameters)
    # load trained network
    with np.load(name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    print("Compiling functions..")
    # visualize convLayers
    conv_w = theano.function([],netLayers['2'].W)
    weights=conv_w()

    ##!!## plot considering 20log?
    ##!!## set min/max to visualize always the same?

    # plot W!
    for i in range(len(weights)):
        plt.subplot(1, len(weights), i+1)
        plt.imshow(np.squeeze(weights[i]), cmap=plt.cm.Reds, interpolation='None', aspect='auto')
    plt.colorbar()
    plt.show()

def trainingEvolution(name):
    'Plot the training evolution: training loss, validation loss and validation accuracy.'

    # load data
    df = pd.read_csv(name+'.training')
    trainingLoss = df['trainingLoss']
    validationLoss = df['validationLoss']
    validationAccuracy = df['validationAccuracy']

    # plot training evolution!
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(range(1,len(trainingLoss)+1,1),trainingLoss, color='red',linestyle='--', marker='o',label="Training Loss")
    plt.hold(True)
    plt.plot(range(1,len(trainingLoss)+1,1),validationLoss,color='blue',linestyle='--', marker='o',label="Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('Validation Accuracy (%)')
    plt.plot(range(1,len(trainingLoss)+1,1),validationAccuracy,color='blue',linestyle='--', marker='o')

    plt.show()

name='./data/results/ballroom_cnn1_46378760430139206020064112940399742791'
visualizeWcnn1(name)
trainingEvolution(name)