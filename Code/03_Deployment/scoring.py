# Prepare the web service definition by authoring
# init() and run() functions. 
def init():
    import os
    #Set backend to tensorflow
    os.environ['KERAS_BACKEND']='tensorflow'
    from keras.models import Sequential
    from keras.layers import Dense
    from keras.models import load_model
    
    # load the model
    global trainedmodel
    network_type= 'unidirectional'
    # network_type= 'bidirectional'
    window_size = 5
    vector_size = 50
    min_count =400
    num_classes = 7 + 1    
    num_layers = 1
    num_epochs = 1
    model_file_path = os.path.join(home_dir,'Models/lstm_{}_model_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_layers, num_epochs, vector_size,window_size, min_count ))

    trainedmodel = load_model(model_file_path)
    
def run(inputstring):
    import numpy
    input1 = numpy.fromstring(inputstring,dtype=float, sep=',').reshape((1,8))
    score=trainedmodel.predict(input1)
    return str(score[0][0])