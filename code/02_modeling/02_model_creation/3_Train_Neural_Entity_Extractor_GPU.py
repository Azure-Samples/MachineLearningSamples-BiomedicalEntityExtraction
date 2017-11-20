
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
# ## Training a Neural Entity Detector using Pubmed Word Embeddings
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
run_logger.log('amlrealworld.BiomedicalEntityExtraction.Train-NeuralEntityExtractor-GPU','true')

#########################################################
#   save_embeddings_to_pickle_file
#   Generate the Embedding Matrix from Parquet files on the Container linked to your Spark Cluster
#########################################################
def download_embedding_parquet_files_from_storage(embedding_full_path, embedding_folder_name, num_parquet_files = 1000):
    
    timestart = datetime.datetime.now()

    #Azure Blob Storage account information
    storage_account_name = '<storage_account_name>'
    storage_account_key ='<storage_account_key>'
    storage_container_name ='<storage_container_name>'       
   
    block_blob_service = BlockBlobService(account_name = storage_account_name, account_key = storage_account_key)          

    if os.path.exists(embedding_full_path):
        shutil.rmtree(embedding_full_path)

    os.makedirs(embedding_full_path)

    count = 0
    generator = block_blob_service.list_blobs(storage_container_name)
    for blob in generator:      
        if embedding_folder_name in blob.name and blob.name.endswith(".parquet"):              
            count = count +1
            filename = blob.name.split("/")[-1]
            block_blob_service.get_blob_to_path(storage_container_name, blob.name, os.path.join(embedding_full_path,filename))      
            print("downloading {}".format(blob.name))

        if count == num_parquet_files:
          break

    print ("Reading {} parquet files".format(count))
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds() / 60, 2)
    print ("Time taken to execute the download_embedding_parquet_files_from_storage function: " + str(timedelta) + " mins")
    
    print ("Done")  

#########################################################
#   save_embeddings_to_pickle_file
#########################################################
def save_embeddings_to_pickle_file(embedding_full_path,embedding_pickle_file, embed_vector_size):
        import pandas 
        import datetime
        timestart = datetime.datetime.now()

        print ("Embedding vector size =", embed_vector_size)      
        Word2Vec_Model = {}

        print("Reading the Parquet embedding files .... {}".format(embedding_full_path))
        files = os.listdir(embedding_full_path)
        for index, filename in enumerate(files):
            if "part" in filename:        
                parquet_file_path = os.path.join(embedding_full_path,filename)
                print("reading {}".format(parquet_file_path))

                try:
                    pfile = fastparquet.ParquetFile(parquet_file_path) 
                    # convert to pandas dataframe
                    df =  pfile.to_pandas()            
            
                    #print(df.head())    
                    arr = list(df.values)                 
                    for ind, vals in enumerate(arr):
                        word = vals[0]
                        word_vec = vals[1:embed_vector_size+1]
                        word_vec = np.array(word_vec)
                        Word2Vec_Model[word] = word_vec.astype('float32')
                except:
                    print("Skip {}".format(filename))
                
        #save the embedding matrix into a pickle file
        print("save the embedding matrix of {} entries into a pickle file".format(len(Word2Vec_Model)))
        pickle.dump(Word2Vec_Model, open(embedding_pickle_file, "wb")) 
        
        timeend = datetime.datetime.now()
        timedelta = round((timeend-timestart).total_seconds() / 60, 2)
        print ("Time taken to execute the save_embeddings_to_pickle_file function: " + str(timedelta) + " mins")
        print ("Done.")


################################################################################### 
#  Train the network on the prepared data and obtain the predictions on the test set
###################################################################################
def main():
    print("Running on BIO-NLP data\n\n")
    
	##########################
    #specify the actions that you would like to execute
    b_download_embedding_files =False
    b_train = True 
    b_evaluate = True
    b_score = False
	##########################

    #Specify the path where to store the downloaded files    

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

    # Define the data files 
    data_folder = os.path.join("sample_data","drugs_and_diseases")
    train_file_path = os.path.join(data_folder, "Drug_and_Disease_train.txt")
    test_file_path = os.path.join(data_folder, "Drug_and_Disease_test.txt")    
    data_file_path = os.path.join(data_folder, "unlabeled_test_sample.txt")    
    resources_pickle_file = os.path.join(home_dir, "models", "resources.pkl")
    embedding_pickle_file = os.path.join(home_dir, "models", "w2vmodel_pubmed_vs_{}_ws_{}_mc_{}.pkl" \
            .format(embed_vector_size, window_size, min_count))
    print("embedding_pickle_file= {}".format(embedding_pickle_file))

    if b_download_embedding_files == True:
        #Specify the string to look for in blob names from your container
        embedding_folder_name = "word2vec_pubmed_model_vs_{}_ws_{}_mc_{}_parquet_files".\
            format(embed_vector_size, window_size, min_count)
        print("embedding_folder_name= {}".format(embedding_folder_name))

        embedding_full_path = os.path.join(home_dir, "models", embedding_folder_name)
        print("embedding_full_path= {}".format(embedding_full_path))                
    
        #download the parquet files from Blob storage
        download_embedding_parquet_files_from_storage(embedding_full_path, embedding_folder_name, num_parquet_files= 1000)              

        save_embeddings_to_pickle_file(embedding_full_path, embedding_pickle_file, embed_vector_size)        
        print("Done")

    
    # The hyperparameters of the LSTM trained model         
    #network_type= 'unidirectional'
    network_type= 'bidirectional'
    #embed_vector_size = 50    
    num_layers = 2
    num_hidden_units = 150
    num_epochs = 10
    batch_size = 50
    dropout = 0.2
    reg_alpha = 0.0

    model_file_path = os.path.join(home_dir,'models','lstm_{}_model_units_{}_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_hidden_units, num_layers,  num_epochs, embed_vector_size, window_size, min_count))    
    
    K.clear_session()
    with K.get_session() as sess:        
        K.set_session(sess)
        graphr = K.get_session().graph
        with graphr.as_default():                        

            if b_train == True:
                print("Training the model... num_epochs = {}, num_layers = {}, num_hidden_units = {}".\
                      format(num_epochs, num_layers,num_hidden_units))

                reader = DataReader() 
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)
               
                entityExtractor.train (train_file_path, \
                    output_resources_pickle_file = resources_pickle_file, \
                    network_type = network_type, \
                    num_epochs = num_epochs, \
                    batch_size = batch_size, \
                    dropout = dropout, \
                    reg_alpha = reg_alpha, \
                    num_hidden_units = num_hidden_units, \
                    num_layers = num_layers)                

                #Save the model
                entityExtractor.save(model_file_path)

            if b_evaluate == True:
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
               
            if b_score == True:
                print("Starting the model prediction ...")

                reader = DataReader(input_resources_pickle_file = resources_pickle_file) 
                entityExtractor = EntityExtractor(reader)
                
                 #load the model
                print("Loading the model from file {} ...".format(model_file_path))
                entityExtractor.load(model_file_path)
                entityExtractor.print_summary()

                predicted_tags = entityExtractor.predict_2(data_file_path)
                
                if not os.path.exists(os.path.join(home_dir, "output")):
                    os.makedirs(os.path.join(home_dir, "output"))

                output_prediction_file = os.path.join(home_dir, "output", "prediction_output.tsv")
                with open(output_prediction_file, 'w') as f:
                    for ind, line in enumerate(predicted_tags):
                        f.write("{}\t{}\n".format(ind,line))                                
                
    K.clear_session()
    K.set_session(None)
    print("Done.")     
   
if __name__ == "__main__":
    main()

