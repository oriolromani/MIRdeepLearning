#!/usr/bin/env python

import time, random, csv
import numpy as np

import theano, lasagne
import theano.tensor as T

import load_datasets as loadData
import buildArchitecture as buildArch

def main(parameters):
    def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
        assert len(inputs) == len(targets)
        if shuffle:
            indices = np.arange(len(inputs))
            np.random.shuffle(indices)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    print("Loading data..")
    parameters['numOutputNeurons'], X_train, y_train, X_val, y_val, X_test, y_test = loadData.load_dataset(parameters)

    print("Building network..")
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    network,netLayers,parameters=buildArch.buildNet(input_var,parameters)

    print("Compiling functions..")

    def computeLoss(prediction, target_var, parameters):
        if parameters['cost']=='crossentropy':
            loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        elif parameters['cost']=='squared_error':
            loss = lasagne.objectives.squared_error(prediction, target_var)
        loss = loss.mean()
        return loss

    # define training functions
    prediction = lasagne.layers.get_output(network)
    loss=computeLoss(prediction, target_var, parameters)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=parameters['lr'], momentum=parameters['momentum'])

    # define testing/val functions
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss=computeLoss(test_prediction, target_var, parameters)
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var), dtype=theano.config.floatX)

    # compile training and test/val functions
    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    print("Training..")
    hash = random.getrandbits(128)
    trainLoss_ans=np.inf
    for epoch in range(parameters['num_epochs']):
        # training set
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, parameters['batchSize'], shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # validation set
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val, parameters['batchSize'], shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # output
        print("Epoch {} of {} took {:.3f}s".format(
            epoch + 1, parameters['num_epochs'], time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))

        #################  JUST STORING OUTPUTS INTO FILES FOR TRACKING ##################
        name='./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)
        # save the best model
        if train_err/train_batches<trainLoss_ans: # [DOUBT] train loss or validation accuracy?
            np.savez(name, *lasagne.layers.get_all_param_values(network))
            res = open('./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)+'.result', 'w')
            res.write("Epoch {} of {} took {:.3f}s\n".format(epoch + 1, parameters['num_epochs'], time.time() - start_time))
            res.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
            res.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
            res.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches * 100))
            res.close()
            trainLoss_ans=train_err/train_batches
        # save parameters
        if epoch==0:
            param = open('./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)+'.param', 'w')
            for key, value in parameters.iteritems():
                param.write('-'+str(key))
            param.write('\n')
            for key, value in parameters.iteritems():
                param.write('-'+str(value))
            param.write('\n')
            param.close()
            tr = open('./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)+'.training', 'w')
            tr.write('epoch,trainingLoss,validationLoss,validationAccuracy\n')
            tr.close()
        # save training evolution
        tr = open('./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)+'.training', 'a')
        tr.write(str(epoch)+','+str(train_err/train_batches)+','+str(val_err / val_batches)+','+str(val_acc / val_batches * 100)+'\n')
        tr.close()
        ##################################################################################

    print("Testing with the best model..")
    # load best model
    with np.load(name+'.npz') as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    # test it!
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, y_test, parameters['batchSize'], shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1

    # output
    print("Final results:")
    print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
    print("  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100))

    #################  JUST STORING OUTPUTS INTO FILES FOR TRACKING ##################
    res = open('./data/results/'+parameters['dataset']+'_'+parameters['type']+'_'+str(hash)+'.result', 'a')
    res.write("\nFinal results:\n")
    res.write("  test loss:\t\t\t{:.6f}\n".format(test_err / test_batches))
    res.write("  test accuracy:\t\t{:.2f} %\n".format(test_acc / test_batches * 100))
    res.close()
    ##################################################################################