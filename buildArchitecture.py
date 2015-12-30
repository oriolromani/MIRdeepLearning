import lasagne

def buildNet(input_var, parameters):
    if parameters['type']=='cnn1':
        return cnn1(input_var, parameters)
    else:
        print 'Architecture NOT supported'

def cnn1(input_var,parameters):

    parameters['filter_size']=(12,8)
    parameters['num_filters']=15
    parameters['nonlinearity']=lasagne.nonlinearities.rectify
    parameters['W_init']=lasagne.init.GlorotUniform()
    parameters['pool_size']=(2, 1)
    parameters['dropout_p']=.5
    parameters['num_dense_units']=200

    network={}
    network["1"] = lasagne.layers.InputLayer(shape=(None, int(parameters['numChannels']), int(parameters['melBands']), int(parameters['inputFrames'])),input_var=input_var)
    network["2"] = lasagne.layers.Conv2DLayer(network["1"], num_filters=parameters['num_filters'], filter_size=parameters['filter_size'],nonlinearity=parameters['nonlinearity'],W=parameters['W_init'])
    network["3"] = lasagne.layers.MaxPool2DLayer(network["2"], pool_size=parameters['pool_size'])
    network["4"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["3"], p=parameters['dropout_p']),num_units=parameters['num_dense_units'],nonlinearity=parameters['nonlinearity'])
    network["5"] = lasagne.layers.DenseLayer(lasagne.layers.dropout(network["4"], p=parameters['dropout_p']),num_units=int(parameters['numOutputNeurons']),nonlinearity=lasagne.nonlinearities.softmax)

    return network["5"],network,parameters