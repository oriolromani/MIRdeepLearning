import MIRdl

# Input data parameters
parameters={}
parameters['dataset'] = 'ballroom'
parameters['frameSize'] = 2048
parameters['hopSize'] = 1024
parameters['numChannels'] = 1
parameters['windowType'] = 'blackmanharris62'
parameters['melBands'] = 40
parameters['inputFrames'] = 80
parameters['errorCode'] = 999
parameters['inputNorm'] = 'energy20Log'

# Deep Learning architecture
parameters['type'] = 'cnn1'
parameters['task'] = 'classification'

# Training parameters
parameters['trainSplit'] = 0.75
parameters['testSplit'] = 0.15
parameters['valSplit'] = 1-parameters['trainSplit']-parameters['testSplit']
parameters['num_epochs'] = 10
parameters['batchSize'] = 500
parameters['lr'] = 0.02
parameters['momentum'] = 0.9
parameters['cost']='crossentropy'

MIRdl.main(parameters)