
import numpy as np
import logging, sys, json
import timeit as t
from keras.models import load_model
from DataPreparation import Data_Preparation
from sklearn.externals import joblib

logger = logging.getLogger("stmt_logger")
ch = logging.StreamHandler(sys.stdout)
logger.addHandler(ch)

# vectors_file='embeddings_SSWE_Basic_Keras_w_TF.tsv'
raw_embedding_file='embeddings_SSWE_Basic_Keras_w_CNTK.tsv'
vectors_file='pickle_'+raw_embedding_file
trainedModelFile="evaluation_SSWE_logistic"

trainedModel = None
mem_after_init = None
labelLookup = None
topResult = 3


def init():
    """ Initialise SD model
    """
    global trainedModel, labelLookup, mem_after_init, vector_size, reader,vectors_file

    start = t.default_timer()
    
    vector_size = 50
       
    reader = Data_Preparation(vectors_file)
    
    # Load model and load the model from brainscript (3rd index)
    try:
        trainedModel = joblib.load(trainedModelFile)
    except:
        trainedModel=load_model(trainedModelFile)
        pass
    end = t.default_timer()

    loadTimeMsg = "Model loading time: {0} ms".format(round((end-start)*1000, 2))
    logger.info(loadTimeMsg)

    
def run(content):
    """ Classify the input using the loaded model
    """
    global trainedModel
    start = t.default_timer()
    # Generate Predictions

    line=content.upper()
    test_x=reader.get_sentence_embedding(['<BOS> '+line+' <EOS>'])
    predictions = trainedModel.predict(test_x[0].flatten().reshape(1,150))

    y_pred = np.argmax(predictions, axis=1)

    y_pred_pos = predictions[:,1][0]
    
    text_annotated=''
    if y_pred_pos>=0.5:
        text_annotated+="<b> <font size ='2' color ='green'> Positive {} </font></b>".format(y_pred_pos)
    else:
        text_annotated+="<b> <font size ='2' color ='red'> Negative {} </font></b>".format(y_pred_pos)
    print(y_pred_pos)
    end = t.default_timer()

    logger.info("Predictions took {0} ms".format(round((end-start)*1000, 2)))

    
    return (text_annotated, 'Computed in {0} ms'.format(round((end-start)*1000, 2)))