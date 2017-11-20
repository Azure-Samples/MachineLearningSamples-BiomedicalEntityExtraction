
# coding: utf-8
'''
if you are planning to run this script locally from the DSVM, 
you have to install the dependencies listed in the conda_dependencies.yml file manually.

From Azure ML CLI, run the following commands:

conda install tensorflow-gpu
conda install nltk

pip install h5py
pip install matplotlib
pip install fastparquet
pip install keras
pip install azure-storage

'''
# ## Testing the entity extraction model 
from azure.storage.blob import BlockBlobService
import numpy as np
import datetime
import os
import shutil
import fastparquet
import pickle
import tensorflow
import h5py
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tensorflow.device('/gpu:0')
import nltk 
nltk.download('popular')
from keras.models import load_model
import keras.backend as K
from DataReader import DataReader
from EntityExtractor import EntityExtractor

from azureml.logging import get_azureml_logger

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.BiomedicalEntityExtraction.Test-EntityExtractor-GPU','true')
    
################################################################################### 
#  Load the network and obtain the predictions on the test set
###################################################################################
def main():
    print("Running on BIO-NLP data\n\n")    
    
    from sys import platform
    if platform == "win32":
        home_dir = "C:\\dl4nlp"
    else:
        home_dir = os.path.join(os.path.expanduser('~'), "dl4nlp")

    print("home_dir = {}".format(home_dir))   
   
    # The hyper-parameters of the word embedding trained model 
    window_size = 5
    embed_vector_size = 50
    min_count =400            
         
    data_folder = os.path.join("sample_data","drugs_and_diseases")
    
    test_file_path = os.path.join(data_folder, "Drug_and_Disease_test.txt")       
    
    resources_pickle_file = os.path.join(home_dir, "models", "resources.pkl")

    # The hyper-parameters of the LSTM trained model        
    #network_type= 'unidirectional'
    network_type= 'bidirectional'
    #embed_vector_size = 50
    num_classes = 7 + 1
    max_seq_length = 613
    num_layers = 2
    num_hidden_units = 150
    num_epochs = 10
    batch_size = 50
    dropout = 0.2
    reg_alpha = 0.0
    
    print("Initializing data...")                  

    model_file_path = os.path.join(home_dir,'models','lstm_{}_model_units_{}_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_hidden_units, num_layers,  num_epochs, embed_vector_size, window_size, min_count))    
  
    K.clear_session()
    with K.get_session() as sess:        
        K.set_session(sess)
        graphr = K.get_session().graph
        with graphr.as_default(): 
                        
            # Evaluate the model
            print("Evaluating the model...")

            reader = DataReader(input_resources_pickle_file = resources_pickle_file)   
            entityExtractor = EntityExtractor(reader)

            #load the model
            print("Loading the model from file {} ...".format(model_file_path))
            entityExtractor.load(model_file_path)
            entityExtractor.print_summary()                              
                
            if not os.path.exists(os.path.join(home_dir, "output")):
                os.makedirs(os.path.join(home_dir, "output"))

            # make sure that the input test data file is in IOB format
            output_prediction_file = os.path.join(home_dir, "output", "prediction_output.tsv")

            evaluation_report, confusion_matrix = entityExtractor.evaluate_model(test_file_path, output_prediction_file)
            print(evaluation_report) 
            print(confusion_matrix) 
                                              
            #########################################################
            # from the commmand line interface, 
            # (1) change directory to \code\02_modeling\03_model_evaluation
            # (2) run the following perl evaluation script 
            # "C:\Program Files\Git\usr\bin\perl.exe" Drug_and_Disease_eval.pl ..\..\..\sample_data\drugs_and_diseases\Drug_and_Disease_test.txt C:\dl4nlp\output\prediction_output.tsv
            #########################################################
    K.clear_session()
    K.set_session(None)
    print("Done.")     
   
if __name__ == "__main__":
    main()


