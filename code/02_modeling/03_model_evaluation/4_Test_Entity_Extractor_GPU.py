
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
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tensorflow.device('/gpu:0')
import nltk 
nltk.download('popular')
from keras.models import load_model
import keras.backend as K
from DataReader import DataReader
from EntityExtractor import EntityExtractor



############################################################################# 
#  Load the network on the prepared data and obtain the predictions on the test set
#############################################################################
if __name__ == "__main__":
    print("Running on BIO-NLP data\n\n")    
    #Specify the path where to store the downloaded files    
    home_dir = "C:\\dl4nlp" 
    print("home_dir = {}".format(home_dir))
   

    #Azure BLOB Storage account information
    storage_account_name = '76f8577bf451dsvm'
    storage_account_key ='5DPDh+p3Xbg9BfS9d/OSrtQ/Utrat1Rr/NRrGU+x3cRYPZYi6B92WEWUIkM28Z8cGRsRz0cuSGb2mjyBCB0QXg=='
    storage_container_name ='dl4nlp-container'    
    
    window_size = 5
    embed_vector_size = 50
    min_count =1000
    #download_embedding_parquet_files_from_storage()
    embedding_pickle_file = save_embeddings_to_pickle_file()
    
    #embedding_pickle_file = os.path.join(home_dir, "Models/w2vmodel_pubmed_vs_{}_ws_{}_mc_{}.pkl" \
    #        .format(embed_vector_size, window_size, min_count))

    # Read the data
    #train_file_local_path = os.path.join(home_dir, train_file_relative_path)
    #test_file_local_path = os.path.join(home_dir, test_file_relative_path)
    #data_file_local_path = os.path.join(home_dir, data_file_relative_path)
    #train_file_local_path, test_file_local_path = download_data_from_storage()  
    data_folder = os.path.join("sample_data","drugs_and_diseases/")
    train_file_local_path = os.path.join(data_folder, "Drug_and_Disease_train.txt")
    test_file_local_path = os.path.join(data_folder, "Drug_and_Disease_test.txt")    
    #data_file_relative_path= os.path.join(data_folder, "unlabeled_test_sample.txt")

    tag_to_idx_map_file = os.path.join(home_dir, "Models", "tag_map.tsv")

    # Train the model        
    #network_type= 'unidirectional'
    network_type= 'bidirectional'
    #embed_vector_size = 50
    num_classes = 7 + 1
    max_seq_length = 613
    num_layers = 2
    num_hidden_units = 300
    num_epochs = 10
    
    print("Initializing data...")                  

    model_file_path = os.path.join(home_dir,'Models','lstm_{}_model_units_{}_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_hidden_units, num_layers,  num_epochs, embed_vector_size, window_size, min_count))
    
    #mode = 'train'
    mode = 'evaluate'
    #mode = 'score'
    K.clear_session()
    with K.get_session() as sess:        
        K.set_session(sess)
        graphr = K.get_session().graph
        with graphr.as_default():                        

            if mode == 'train':
                print("Training the model... num_epochs = {}, num_layers = {}".format(num_epochs, num_layers))

                reader = DataReader(num_classes, vector_size =embed_vector_size) 
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)
                entityExtractor.train(train_file_local_path, network_type, num_epochs, num_hidden_units, num_layers)    
                entityExtractor.save_tag_map(tag_to_idx_map_file)

                #Save the model
                entityExtractor.save(model_file_path)
            elif mode == 'evaluate':
                # Evaluate the model
                print("Evaluating the model...")

                reader = DataReader(num_classes, max_seq_length, tag_to_idx_map_file, vector_size=embed_vector_size)   
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)

                #load the model
                print("Loading the model from file {} ...".format(model_file_path))
                entityExtractor.load(model_file_path)
                entityExtractor.print_summary()                
                
                confusion_matrix = entityExtractor.evaluate_model(test_file_local_path)
                print(confusion_matrix)
            elif mode == 'score':
                print("Starting the model prediction ...")

                reader = DataReader(num_classes, max_seq_length, tag_to_idx_map_file, vector_size=embed_vector_size) 
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)
                
                 #load the model
                print("Loading the model from file {} ...".format(model_file_path))
                entityExtractor.load(model_file_path)
                entityExtractor.print_summary()

                predicted_tags = entityExtractor.predict_2(data_file_local_path)
                if not os.path.exists("C:\dl4nlp\output"):
                    os.makedirs("C:\dl4nlp\output")

                with open('C:\dl4nlp\output\prediction.out', 'w') as f:
                    for ind, line in enumerate(predicted_tags):
                        f.write("{}\t{}\n".format(ind,line))

            else:
                print("undefined mode")                        
                
    K.clear_session()
    K.set_session(None)
    print("Done.")     
    
'''

# #### Step 5 Generate the output of the model in the correct format for evaluation

# In[11]:

file1 = open("Pubmed_Output.txt")
file2 = open("Drugs_and_Diseases//test.txt")
target = open("Drugs_and_Diseases//eval2.txt", "w")

list1 = []
list2 = []

for line in file1:
    list1.append(line)
    
for line in file2:
    list2.append(line)
    
for ind, line in enumerate(list2):
    x = line.split("\t")
    if len(x) == 1:
        target.write("\n")
    else:
        target.write(x[0])
        target.write("\t")
        if list1[ind] == "NONE":
            target.write("O")
        else:
            target.write(list1[ind])
    ind += 1
    
file1.close()
file2.close()
target.close()


# #### Evaluate the model predictions on the test data

# In[12]:

get_ipython().system(u'./Drugs_and_Diseases/evalD_a_D.pl Drugs_and_Diseases/eval2.txt Drugs_and_Diseases/test.txt #with Embedding Layer')

'''