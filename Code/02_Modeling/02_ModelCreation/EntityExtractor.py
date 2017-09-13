# %%writefile Entity_Extractor.py
from keras.preprocessing import sequence
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers.core import Activation
from keras.regularizers import l2
from keras.layers.wrappers import TimeDistributed
from keras.layers.wrappers import Bidirectional
from keras.layers.normalization import BatchNormalization
from keras.layers import Embedding
from keras.layers.core import Dropout
import numpy as np
import pandas as pd
import sys
import keras.backend as K
from sklearn.metrics import confusion_matrix, classification_report

# For reproducibility
np.random.seed(42)

class EntityExtractor:

    def __init__ (self, reader, embeddings_file_path):
        
        self.reader = reader
        self.model = None
        #self.all_X_train, self.all_Y_train, self.all_X_test, self.all_Y_test, self.wordvecs = \
        #    reader.get_data()

        self.wordvecs = self.reader.load_embedding_lookup_table(embeddings_file_path)

        #self.train_X = self.all_X_train
        #self.train_Y = self.all_Y_train
        
        #self.test_X = self.all_X_test
        #self.test_Y = self.all_Y_test

    def save_tag_map (self, filepath):     
         with open(filepath, 'w') as f:
            for tag in self.reader.tag_to_vector_map.keys():
                try:
                    tag_index = self.reader.tag_to_vector_map[tag].index(1)
                except:
                    continue

                f.write("{}\t{}\n".format(tag, tag_index));

    def load (self, filepath):
        self.model = load_model(filepath)
        
    def save (self, filepath):
        self.model.save(filepath)

    def print_summary (self):
        print(self.model.summary())        
   
    def train (self, train_file, network_type = 'unidirectional', \
               num_epochs = 1, batch_size = 50, dropout = 0.2, reg_alpha = 0.0, \
               num_hidden_units = 150, num_layers = 1):
        
        train_X, train_Y = self.reader.read_and_parse_training_data(train_file)       

#         self.train_X = self.all_X_train
#         self.train_Y = self.all_Y_train
        
#         self.test_X = self.all_X_test
#         self.test_Y = self.all_Y_test

        print("Data Shape: ")
        print(train_X.shape)
        print(train_Y.shape)        
        
        dropout = 0.2                
                
        self.model = Sequential()        
        self.model.add(Embedding(self.wordvecs.shape[0], self.wordvecs.shape[1], \
                                 input_length = train_X.shape[1], \
                                 weights = [self.wordvecs], trainable = False))
        
        if network_type == 'unidirectional':
            # uni-directional LSTM
            self.model.add(LSTM(num_hidden_units, return_sequences = True))
        else:
            # bi-directional LSTM
            self.model.add(Bidirectional(LSTM(num_hidden_units, return_sequences = True)))
        
        self.model.add(Dropout(dropout))

        for i in range(1, num_layers):
            if network_type == 'unidirectional':
                # uni-directional LSTM
                self.model.add(LSTM(num_hidden_units, return_sequences = True))
            else:
                # bi-directional LSTM
                self.model.add(Bidirectional(LSTM(num_hidden_units, return_sequences = True)))
        
            self.model.add(Dropout(dropout))

        self.model.add(TimeDistributed(Dense(train_Y.shape[2], activation='softmax')))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        print(self.model.summary())

        self.model.fit(train_X, train_Y, epochs = num_epochs, batch_size = batch_size)
                
              
    def score_model(self, data_file):
        import json
        from collections import OrderedDict as odict

        feat_vector_list, word_seq_list, num_tokens_list = self.reader.read_and_parse_unlabeled_data(data_file)
        print("Data Shape: ")        
        print(feat_vector_list.shape)
        # the output is a list of JSON strings
        predicted_tags= []
        ind = 0
        for feat_vector, word_seq, num_tokens in zip(feat_vector_list, word_seq_list, num_tokens_list):
            prob_dist = self.model.predict(np.array([feat_vector]), batch_size=1)[0]
            pred_tags = self.reader.decode_prediction_sequence(prob_dist)
            pred_dict = odict(zip(word_seq[-num_tokens:], pred_tags[-num_tokens:]))
            pred_str = json.dumps(pred_dict) 
            predicted_tags.append(pred_str)
            ind += 1
            ### To see Progress ###
            if ind % 500 == 0: 
                print("Sentence" + str(ind))
        
        #predicted_tags = np.array(predicted_tags)
        return predicted_tags
    
    def evaluate_model(self, test_file):
        target = open("Pubmed_Output.txt", 'w')
        
        test_X, test_Y = self.reader.read_and_parse_test_data(test_file)
        
        print("Data Shape: ")        
        print(test_X.shape)
        print(test_Y.shape)
        
        predicted_tags= []
        test_data_tags = []
        ind = 0
        for x,y in zip(test_X, test_Y):
            tags = self.model.predict(np.array([x]), batch_size=1)[0]
            pred_tags = self.reader.decode_prediction_sequence(tags)
            test_tags = self.reader.decode_prediction_sequence(y)
            ind += 1
            ### To see Progress ###
            if ind%500 == 0: 
                print("Sentence" + str(ind))

            pred_tag_wo_none = []
            test_tags_wo_none = []
            
            for index, test_tag in enumerate(test_tags):
                if test_tag != "NONE":
                    if pred_tags[index] == "B-Chemical":
                        pred_tag_wo_none.append("B-Drug")
                    elif pred_tags[index] == "I-Chemical":
                        pred_tag_wo_none.append("I-Drug")
                    elif pred_tags[index] == 'None':
                        pred_tag_wo_none.append('O')
                    else:
                        pred_tag_wo_none.append(pred_tags[index])
                        
                    if test_tag == "B-Chemical":
                        test_tags_wo_none.append("B-Drug")
                    elif test_tag == "I-Chemical":
                        test_tags_wo_none.append("I-Drug")
                    else:                        
                        test_tags_wo_none.append(test_tag)
            
            for wo in pred_tag_wo_none:
                target.write(str(wo))
                target.write("\n")
            target.write("\n")
            
            for i,j in zip(pred_tags, test_tags):
                if i != "NONE" and j != "NONE":
                    test_data_tags.append(j)
                    predicted_tags.append(i)

        target.close()
        
        predicted_tags = np.array(predicted_tags)
        test_data_tags = np.array(test_data_tags)
        print(classification_report(test_data_tags, predicted_tags))

        simple_conf_matrix = confusion_matrix(test_data_tags,predicted_tags)
        all_tags = sorted(list(set(test_data_tags)))
        conf_matrix = pd.DataFrame(columns = all_tags, index = all_tags)
        for x,y in zip(simple_conf_matrix, all_tags):
            conf_matrix[y] = x
        conf_matrix = conf_matrix.transpose()
        
        return conf_matrix