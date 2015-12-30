import sys, os, pickle
import numpy as np

from essentia.standard import *

def load_dataset(parameters):
    if parameters['dataset']=='ballroom':
        if parameters['task'] == 'classification':
            return ballroom_classification(parameters)

    elif parameters['dataset']=='GTZAN':
        if parameters['task'] == 'classification':
            return GTZAN_classification(parameters)

    else:
        print('Dataset NOT supported')

def normalization(spect,inputNorm):

    if inputNorm=='energy':
        E=sum(sum(np.power(spect, 2)))/len(spect)
        spect=spect/E

    elif inputNorm=='energy20Log':
        E=sum(sum(np.power(spect, 2)))/len(spect)
        spect=spect/E
        spect = 20*np.log10(spect+1)

    elif inputNorm=='energyLog':
        E=sum(sum(np.power(spect, 2)))/len(spect)
        spect=spect/E
        spect = np.log10(spect+1)

    elif inputNorm=='log':
        spect = np.log10(spect+1)

    elif inputNorm=='None':
        print 'No normalization!'

    else:
        print 'This normalization does not exist!'

    ##!## remove silences ?
    ##!## log10(x+1) by itself avoids nans. No need of more stuff such as: spect[spect == -np.inf]= 0 or np.finfo(float).eps
    ##!## mean 0 variance 1 ?

    return spect

def GTZAN_classification(parameters):

    pickleFile='./data/preloaded/'
    for key, value in parameters.iteritems():
        pickleFile=pickleFile+'_'+str(value)
    pickleFile=pickleFile+'.pickle'

    if os.path.exists(pickleFile):

        print "-- Loading pre-computed spectrograms.."

        with open(pickleFile) as f:
            numOutputNeurons, X_train, X_test, X_val, y_train, y_test, y_val = pickle.load(f)

    else:

        dir= './data/datasets/GTZAN'
        dict = {'blues':0,'classical':1,'country':2,'disco':3,'hiphop':4,'jazz':5,'metal':6,'pop':7,'reggae':8,'rock':9}

        numInputs=16000
        # if not known: numInputs=countNumInputs(dir,parameters)

        numOutputNeurons, X_train, y_train, X_val, y_val, X_test, y_test = formatAudioClassification(dir,dict,numInputs,parameters)

        # Saving the objects:
        with open(pickleFile, 'w') as f:
            pickle.dump([numOutputNeurons, X_train, X_test, X_val, y_train, y_test, y_val], f)

    return numOutputNeurons, X_train, y_train, X_val, y_val, X_test, y_test

def ballroom_classification(parameters):

    pickleFile='./data/preloaded/'
    for key, value in parameters.iteritems():
        pickleFile=pickleFile+'_'+str(value)
    pickleFile=pickleFile+'.pickle'

    if os.path.exists(pickleFile):

        print "-- Loading pre-computed features.."

        with open(pickleFile) as f:
            numOutputNeurons, X_train, X_test, X_val, y_train, y_test, y_val = pickle.load(f)

    else:

        dir= './data/datasets/Ballroom/BallroomData'
        dict = {'ChaChaCha':0,'Samba':1,'Quickstep':2,'VienneseWaltz':3,'Tango':4,'Jive':5,'Waltz':6,'Rumba':7}

        numInputs=11629
        # if not known: numInputs=countNumInputs(dir,parameters)

        numOutputNeurons, X_train, y_train, X_val, y_val, X_test, y_test = formatAudioClassification(dir,dict,numInputs,parameters)

        # Saving the objects:
        with open(pickleFile, 'w') as f:
            pickle.dump([numOutputNeurons, X_train, X_test, X_val, y_train, y_test, y_val], f)

    return numOutputNeurons, X_train, y_train, X_val, y_val, X_test, y_test

def formatAudioClassification(dir,dict,num_inputs,parameters):

    D=np.zeros(num_inputs*parameters['melBands']*parameters['inputFrames'],dtype=np.float32).reshape(num_inputs,1,parameters['melBands'],parameters['inputFrames'])
    A=np.zeros(num_inputs,dtype=np.uint8)+parameters['errorCode']
    count=0
    for root, dirs, files in os.walk(dir):
        for annotation in dirs:
            for r, ds, fs in os.walk(root+'/'+annotation):
                for f in fs:
                    spect = computeMelSpectrogram(root+'/'+annotation+'/'+f,parameters['frameSize'],parameters['hopSize'],parameters['windowType'],parameters['melBands'])
                    spect=normalization(spect,parameters['inputNorm'])
                    for c in chunk(spect,parameters['inputFrames']):
                        D[count][0]=c
                        A[count]=dict[annotation]
                        count=count+1
                    print(root+'/'+annotation+'/'+f)

    D,A=shuffle_in_unison_inplace(D, A)

    cut_train=int(np.floor(parameters['trainSplit']*D.shape[0]))
    cut_test=int(np.floor((parameters['trainSplit']+parameters['testSplit'])*D.shape[0]))
    X_train, X_test, X_val = D[:cut_train], D[cut_train+1:cut_test], D[cut_test+1:]
    y_train, y_test, y_val = A[:cut_train], A[cut_train+1:cut_test], A[cut_test+1:]

    numOutputNeurons=len(set(A)) # number of different clases

    return numOutputNeurons, X_train, y_train, X_val, y_val, X_test, y_test

def countNumInputs(dir,parameters):
        num_inputs=0
        for root, dirs, files in os.walk(dir):
            for annotation in dirs:
                for r, ds, fs in os.walk(root+'/'+annotation):
                    for f in fs:
                        spect = computeMelSpectrogram(root+'/'+annotation+'/'+f,parameters['frameSize'],parameters['hopSize'],parameters['windowType'],parameters['melBands'])
                        num_inputs=num_inputs+len(chunk(spect,80))
                        print num_inputs

def computeMelSpectrogram(file,frameSize,hopSize,windowType,melBands):
    loader = essentia.standard.MonoLoader(filename = file)
    audio = loader()
    w = Windowing(type = windowType)
    spectrum = Spectrum()
    mel = MelBands(numberBands = melBands) # 40 bands!

    melSpec = []
    for frame in FrameGenerator(audio, frameSize = frameSize, hopSize = hopSize):
        melSpec.append(mel(spectrum(w(frame))))
    # we need to convert the list to an essentia.array first (== numpy.array of floats)
    melSpec = essentia.array(melSpec).T

    return melSpec

def chunk(l, n):
    """Yield successive n-sized chunks from l."""
    out=[]
    for i in xrange(0, int(l.shape[1]/n)*n, n):
        out.append(l[:,i:i+n])

    return out

def shuffle_in_unison_inplace(a, b):
            assert len(a) == len(b)
            p = np.random.permutation(len(a))
            return a[p], b[p]