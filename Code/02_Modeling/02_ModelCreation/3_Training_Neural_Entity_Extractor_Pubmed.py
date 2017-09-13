
# coding: utf-8
'''
From Azure ML CLI, run the following commands:

conda install tensorflow-gpu
conda install fastparquet
conda install nltk

'''
# ## Training a Neural Entity Detector using Pubmed Word Embeddings
from DataReader import DataReader
from EntityExtractor import EntityExtractor
import tensorflow
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
tensorflow.device('/gpu:0')
import nltk 
nltk.download('popular')
import os 

# ### Step 1


# #### Generate the Embedding Matrix from Parquet files on the Container linked to your Spark Cluster
def download_embedding_parquet_files_from_storage():
    from azure.storage.blob import BlockBlobService
    import numpy as np
    import datetime

    import os
    import shutil
    import fastparquet
    import pickle
    timestart = datetime.datetime.now()

    block_blob_service = BlockBlobService(account_name = storage_account_name, account_key = storage_account_key)    
    
    #Specify the string to look for in blob names from your container
    embedding_relative_path = "Models/word2vec_pubmed_model_vs_{}_ws_{}_mc_{}_parquet_files".\
        format(vector_size, window_size, min_count)

    embedding_full_path = os.path.join(home_dir, embedding_relative_path)
    print("embedding_full_path= {}".format(embedding_full_path))

    if os.path.exists(embedding_full_path):
        shutil.rmtree(embedding_full_path)

    os.makedirs(embedding_full_path)
            
    num_parquet_files = 0
    generator = block_blob_service.list_blobs(storage_container_name)
    for blob in generator:      
        if embedding_relative_path in blob.name and blob.name.endswith(".parquet"):              
            num_parquet_files = num_parquet_files +1
            filename = blob.name.split("/")[-1]
            block_blob_service.get_blob_to_path(storage_container_name, blob.name, os.path.join(embedding_full_path,filename))      
            
    print ("Reading {} parquet files".format(num_parquet_files))
    timeend = datetime.datetime.now()
    timedelta = round((timeend-timestart).total_seconds() / 60, 2)
    print ("Time taken to execute above cell: " + str(timedelta) + " mins")
            
def save_embeddings_to_pickle_file():
        import pandas 
        import datetime
        timestart = datetime.datetime.now()

        print ("Embedding vector size =", vector_size)
    
        embedding_pickle_file = os.path.join(home_dir, "Models/w2vmodel_pubmed_vs_{}_ws_{}_mc_{}.pkl" \
            .format(vector_size, window_size, min_count))

        Word2Vec_Model = {}

        print("Reading the Parquet embedding files ....")
        files = os.listdir(embedding_full_path)
        for index, filename in enumerate(files):
            if "part" in filename:        
                parquet_file_path = os.path.join(embedding_full_path,filename)
                print("reading {}".format(parquet_file_path))

                try:
                    pfile = fastparquet.ParquetFile(parquet_file_path) 
                    # convert to pandas dataframe
                    df =  pfile.to_pandas()    
        #             df = pandas.read_csv(tsv_full_path, sep='\t')
            
                    #print(df.head())    
                    arr = list(df.values)                 
                    for ind, vals in enumerate(arr):
                        word = vals[0]
                        word_vec = vals[-vector_size:]
                        word_vec = np.array(word_vec)
                        Word2Vec_Model[word] = word_vec.astype('float32')
                except:
                    print("Skip {}".format(filename))
                
        #save the embedding matrix into a pickle file
        print("save the embedding matrix of {} entries into a pickle file".format(len(Word2Vec_Model)))
        pickle.dump(Word2Vec_Model, open(embedding_pickle_file, "wb")) 
        
        timeend = datetime.datetime.now()
        timedelta = round((timeend-timestart).total_seconds() / 60, 2)
        print ("Time taken to execute above cell: " + str(timedelta) + " mins")
        return(embedding_pickle_file)

# #### Copy the Training Data, Testing Data, Evaluation Script to destination location
def download_data_from_storage():
    from azure.storage.blob import BlockBlobService
    import os
    block_blob_service = BlockBlobService(account_name = storage_account_name, account_key = storage_account_key)

    generator = block_blob_service.list_blobs(storage_container_name)

    if not os.path.exists(os.path.join(home_dir, data_folder)):
        os.makedirs(os.path.join(home_dir, data_folder))
    
    local_train_file_path  = os.path.join(home_dir, train_file_path)
    local_test_file_path  = os.path.join(home_dir, test_file_path)
        
    block_blob_service.get_blob_to_path(storage_container_name, train_file_path, local_train_file_path)
    block_blob_service.get_blob_to_path(storage_container_name, test_file_path, local_test_file_path)

    return (local_train_file_path, local_test_file_path)

# #### Step 4 Train the network on the prepared data and obtain the predictions on the test set
# from Data_Preparation2 import Data_Preparation2
# from Entity_Extractor import Entity_Extractor
#import cPickle as cp
from keras.models import load_model
import keras.backend as K
import numpy as np

if __name__ == "__main__":
    print("Running on BIO-NLP data\n\n")    
    #Specify the path where to store the downloaded files    
    home_dir = "C:\\dl4nlp" 
    print("home_dir = {}".format(home_dir))
    data_folder = "Data/Drugs_and_Diseases/"
    train_file_path = os.path.join(home_dir, data_folder, "train_out.txt")
    test_file_path = os.path.join(home_dir, data_folder, "test.txt")    
    
    #Azure BLOB Storage account information
    storage_account_name = '76f8577bf451dsvm'
    storage_account_key ='5DPDh+p3Xbg9BfS9d/OSrtQ/Utrat1Rr/NRrGU+x3cRYPZYi6B92WEWUIkM28Z8cGRsRz0cuSGb2mjyBCB0QXg=='
    storage_container_name ='dl4nlp-container'    
    
    window_size = 5
    vector_size = 50
    min_count =400
    #download_embedding_parquet_files_from_storage()
    #embedding_pickle_file = save_embeddings_to_pickle_file()
    
    embedding_pickle_file = os.path.join(home_dir, "Models/w2vmodel_pubmed_vs_{}_ws_{}_mc_{}.pkl" \
            .format(vector_size, window_size, min_count))

    # Read the data
    #local_train_file_path, local_test_file_path = download_data_from_storage()
    local_train_file_path =  os.path.join(home_dir, "Data/Drugs_and_Diseases/", "train_out.txt")
    local_test_file_path = os.path.join(home_dir, "Data/Drugs_and_Diseases/", "test.txt")
    local_data_file_path= os.path.join(home_dir, "Data/Drugs_and_Diseases/", "unlabeled_test_sample.txt")

    tag_to_idx_map_file = os.path.join(home_dir, "Models/", "tag_map.tsv")

    # Train the model        
    network_type= 'unidirectional'
    # network_type= 'bidirectional'
    vector_size = 50
    num_classes = 7 + 1
    max_seq_length = 613
    num_layers = 1
    num_epochs = 1
    
    print("Initializing data...")                  

    model_file_path = os.path.join(home_dir,'Models/lstm_{}_model_lyrs_{}_epchs_{}_vs_{}_ws_{}_mc_{}.h5'.\
                  format(network_type, num_layers, num_epochs, vector_size, window_size, min_count))
    
    #mode = 'train'
    #mode = 'evaluate'
    mode = 'score'
    K.clear_session()
    with K.get_session() as sess:        
        K.set_session(sess)
        graphr = K.get_session().graph
        with graphr.as_default():                        

            if mode == 'train':
                print("Training the model... num_epochs = {0}, num_layers = {1}".format(num_epochs, num_layers))
                reader = DataReader(num_classes, vector_size =vector_size) 
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)
                entityExtractor.train(local_train_file_path, network_type = 'unidirectional', num_epochs=num_epochs, num_layers=num_layers)    
                entityExtractor.save_tag_map(tag_to_idx_map_file)

                #Save the model
                entityExtractor.save(model_file_path)
            elif mode == 'evaluate':
                # Evaluate the model
                print("Evaluating the model...")

                reader = DataReader(num_classes, max_seq_length=max_seq_length, tag_to_idx_map_file=tag_to_idx_map_file, vector_size =vector_size)   
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)

                #load the model
                print("Loading the model...")
                entityExtractor.load(model_file_path)
                entityExtractor.print_summary()                
                
                confusion_matrix = entityExtractor.evaluate_model(local_test_file_path)
                print(confusion_matrix)
            elif mode == 'score':
                print("Starting the model prediction ...")

                reader = DataReader(num_classes, max_seq_length=max_seq_length, tag_to_idx_map_file=tag_to_idx_map_file, vector_size =vector_size) 
                entityExtractor = EntityExtractor(reader, embedding_pickle_file)
                
                 #load the model
                entityExtractor.load(model_file_path)
                entityExtractor.print_summary()     

                predicted_tags = entityExtractor.score_model(local_data_file_path)
                if not os.path.exists("C:\dl4nlp\output"):
                    os.makedirs("C:\dl4nlp\output")

                with open('C:\dl4nlp\output\prediction.out') as f:
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