# %%writefile DataReader.py
from keras.preprocessing import sequence
import numpy as np
#Pytnon 2
#import cPickle as cpickle

#Pytnon 3
import _pickle as cPickle

class DataReader:

    def __init__ (self, num_classes, max_seq_length=0, tag_to_idx_map_file=None, vector_size = 100):
        # Some constants
        self.DEFAULT_N_CLASSES = num_classes
        self.DEFAULT_N_FEATURES = vector_size
        #self.DEFAULT_MAX_SEQ_LENGTH = seq_length
        
        # Other stuff
        self.wordvecs = None
        self.word_to_ix_map = {}
        self.n_features = 0
        self.n_tag_classes = 0
        self.n_sentences_all = 0
        self.tag_to_vector_map = {}
        self.vector_to_tag_map = {}
        
        if not (tag_to_idx_map_file is None):
            with open(tag_to_idx_map_file, 'r') as f:
                for line in f:
                    tag, tag_id = line.split('\t')
                    tag_id = int(tag_id)
                    if tag_id < 0 or tag_id >= self.DEFAULT_N_CLASSES:
                        continue

                    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype = np.int32)
                    one_hot_vec[tag_id] = 1                    
                    self.tag_to_vector_map[tag] = one_hot_vec                    
                    self.vector_to_tag_map[tuple(one_hot_vec)] = tag

        if max_seq_length > 0:
            self.max_sentence_len_train = max_seq_length

        #self.max_sentence_len_test = 0
        #self.max_sentence_len = 0
        
#         self.all_X_train = []
#         self.all_Y_train = []
#         self.all_X_test = []
#         self.all_Y_test = []
        self.unk_words = []        
        
#         self.read_and_parse_data(train_file, test_file, embeddings_file)
            
#     def get_data (self):
#         return (self.all_X_train, self.all_Y_train, \
#                 self.all_X_test, self.all_Y_test, \
#                 self.wordvecs)
    
    def decode_prediction_sequence (self, pred_seq):
        
        pred_tags = []
        for class_prs in pred_seq:
            class_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)
            class_vec[np.argmax(class_prs)] = 1
            if tuple(class_vec.tolist()) in self.vector_to_tag_map:
                pred_tags.append(self.vector_to_tag_map[tuple(class_vec.tolist())])
            else:
                print(tuple(class_vec.tolist()))
        return pred_tags
    
    def load_embedding_lookup_table (self, embeddings_file):
        
        ###Load the Word2Vec Model###
        print("Loading the W2V model from file {}".format(embeddings_file))
        #W2V_model = cPickle.load(open(embeddings_file, "rb"))
        with open(embeddings_file, 'rb') as f:
            W2V_model = cPickle.load(f, encoding='bytes')                     
            
        vocab = list(W2V_model.keys())       
        
        self.word_to_ix_map = {}
        self.wordvecs = []
        
        ###Create LookUp Table for words and their word vectors###
        print("Creating the lookup table")
        for index, word in enumerate(vocab):
            self.word_to_ix_map[word] = index
            self.wordvecs.append(W2V_model[vocab[index]])
            
           
        self.wordvecs = np.array(self.wordvecs)
        print("Number of entries in the lookup table = {}".format(len(self.wordvecs)))
        self.n_features = len(self.wordvecs[0])
        print("embedding size = {}".format(self.n_features))
        
        # Add a zero vector for the Paddings
        self.wordvecs = np.vstack((self.wordvecs, np.zeros(self.DEFAULT_N_FEATURES)))
        self.zero_vec_pos = self.wordvecs.shape[0] - 1
        
        print("Done")
        return (self.wordvecs)
    
    
    
    ##########################  READ TRAINING DATA  ######################### 
    def read_and_parse_training_data (self, train_file, skip_unknown_words = False):
        
        print("Loading the training data from file {}".format(train_file))
        with open(train_file, 'r') as f_train:
            
            self.n_tag_classes = self.DEFAULT_N_CLASSES
            self.tag_to_vector_map = {}    # For storing one hot vector notation for each Tag
            self.vector_to_tag_map = {} 
            tag_class_id = 0            # Used to put 1 in the one hot vector notation
            raw_data_train = []
            raw_words_train = []
            raw_tags_train = []        

            # Process all lines in the file
            for line in f_train:
                line = line.strip()
                if not line:
                    raw_data_train.append( (tuple(raw_words_train), tuple(raw_tags_train)))
                    raw_words_train = []
                    raw_tags_train = []
                    continue
                
                word, tag = line.split('\t')
                
                raw_words_train.append(word)
                raw_tags_train.append(tag)
                
                if tag not in self.tag_to_vector_map:
                    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)
                    one_hot_vec[tag_class_id] = 1
                    self.tag_to_vector_map[tag] = tuple(one_hot_vec)
                    self.vector_to_tag_map[tuple(one_hot_vec)] = tag
                    tag_class_id += 1
                    
        print("number of training examples = " + str(len(raw_data_train)))
        
        #Adding a None Tag
        one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype = np.int32)
        one_hot_vec[tag_class_id] = 1
        self.tag_to_vector_map['NONE'] = tuple(one_hot_vec)
        self.vector_to_tag_map[tuple(one_hot_vec)] = 'NONE'
        tag_class_id += 1
        
        self.n_sentences_all = len(raw_data_train)

        # Find the maximum sequence length for Training data
        self.max_sentence_len_train = 0
        for seq in raw_data_train:
            if len(seq[0]) > self.max_sentence_len_train:
                self.max_sentence_len_train = len(seq[0])                
        
         ############## Create Train Vectors################
        all_X_train, all_Y_train = [], []
        
        self.unk_words = []
        count = 0
        for word_seq, tag_seq in raw_data_train:  
            
            elem_wordvecs, elem_tags = [], []            
            for ix in range(len(word_seq)):
                w = word_seq[ix]
                t = tag_seq[ix]
                w = w.lower()
                if w in self.word_to_ix_map :
                    count += 1
                    elem_wordvecs.append(self.word_to_ix_map[w])
                    elem_tags.append(self.tag_to_vector_map[t])

                elif "UNK" in self.word_to_ix_map :
                    elem_wordvecs.append(self.word_to_ix_map["UNK"])
                    elem_tags.append(self.tag_to_vector_map[t])
                
                else:
                    w = "UNK"       
                    new_wv = 2 * np.random.randn(self.DEFAULT_N_FEATURES) - 1 # sample from normal distribution
                    norm_const = np.linalg.norm(new_wv)
                    new_wv /= norm_const
                    self.wordvecs = np.vstack((self.wordvecs, new_wv))
                    self.word_to_ix_map[w] = self.wordvecs.shape[0] - 1
                    elem_wordvecs.append(self.word_to_ix_map[w])
                    elem_tags.append(list(self.tag_to_vector_map[t]))

            
            # Pad the sequences for missing entries to make them all the same length
            nil_X = self.zero_vec_pos
            nil_Y = np.array(self.tag_to_vector_map['NONE'])
            pad_length = self.max_sentence_len_train - len(elem_wordvecs)
            all_X_train.append( ((pad_length)*[nil_X]) + elem_wordvecs)
            all_Y_train.append( ((pad_length)*[nil_Y]) + elem_tags)

        all_X_train = np.array(all_X_train)
        all_Y_train = np.array(all_Y_train)
        
        print("UNK WORD COUNT = " + str(len(self.unk_words)))
        print("Found WORDS COUNT = " + str(count))
        print("TOTAL WORDS = " + str(count+len(self.unk_words)))    
        
        print("Done")
        
        return (all_X_train, all_Y_train)
    
    
    ############################################################################## 
    #  READ TEST DATA  
    ############################################################################## 
    def read_and_parse_test_data (self, test_file, skip_unknown_words = False):
        
        print("Loading test data from file {}".format(test_file))
        with open(test_file, 'r') as f_test:
            
            self.n_tag_classes = self.DEFAULT_N_CLASSES
            tag_class_id = 0 
            raw_data_test = []
            raw_words_test = []
            raw_tags_test = []        

            # Process all lines in the file
            for line in f_test:
                line = line.strip()
                if not line:
                    raw_data_test.append( (tuple(raw_words_test), tuple(raw_tags_test)))
                    raw_words_test = []
                    raw_tags_test = []
                    continue
                
                word, tag = line.split('\t') 
                
                #if tag not in self.tag_vector_map:
                #    print("added")
                #    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)
                #    one_hot_vec[tag_class_id] = 1
                #    self.tag_vector_map[tag] = tuple(one_hot_vec)
                #    self.tag_vector_map[tuple(one_hot_vec)] = tag
                #    tag_class_id += 1
                
                raw_words_test.append(word)
                raw_tags_test.append(tag)                
                                    
        print("number of test examples = " + str(len(raw_data_test)))   
        self.n_sentences_all = len(raw_data_test)

        # Find the maximum sequence length for Test Data
        #self.max_sentence_len_test = 0
        #for seq in raw_data_test:
        #    if len(seq[0]) > self.max_sentence_len_test:
        #        self.max_sentence_len_test = len(seq[0])
                
#         #Find the maximum sequence length in both training and Testing dataset
#         self.max_sentence_len = max(self.max_sentence_len_train, self.max_sentence_len_test)               
        
        ########################Create TEST Feature Vectors##########################
        self.all_X_test, self.all_Y_test = [], []

        self.unk_words = []
        count = 0
        for word_seq, tag_seq in raw_data_test:  
            
            if len(word_seq) > self.max_sentence_len_train:
               print("skip the extra words in the long sentence")
               word_seq = word_seq[:self.max_sentence_len_train]
               tag_seq = tag_seq[:self.max_sentence_len_train]

            elem_wordvecs, elem_tags = [], []            
            for ix in range(len(word_seq)):
                w = word_seq[ix]
                t = tag_seq[ix]
                w = w.lower()
                if w in self.word_to_ix_map:
                    count += 1
                    elem_wordvecs.append(self.word_to_ix_map[w])
                    elem_tags.append(self.tag_to_vector_map[t])
                    
                elif "UNK" in self.word_to_ix_map :
                    self.unk_words.append(w)
                    elem_wordvecs.append(self.word_to_ix_map["UNK"])
                    elem_tags.append(self.tag_to_vector_map[t])
                    
                else:
                    self.unk_words.append(w)
                    w = "UNK"
                    self.word_to_ix_map[w] = self.wordvecs.shape[0] - 1
                    elem_wordvecs.append(self.word_to_ix_map[w])
                    elem_tags.append(self.tag_to_vector_map[t])
                
            # Pad the sequences for missing entries to make them all the same length
            nil_X = self.zero_vec_pos
            nil_Y = np.array(self.tag_to_vector_map['NONE'])
            pad_length = self.max_sentence_len_train - len(elem_wordvecs)
            all_X_test.append( ((pad_length)*[nil_X]) + elem_wordvecs)
            all_Y_test.append( ((pad_length)*[nil_Y]) + elem_tags)

        all_X_test = np.array(all_X_test)
        all_Y_test = np.array(all_Y_test)
        
        print("UNK WORD COUNT = " + str(len(self.unk_words)))
        print("Found WORDS COUNT = " + str(count))
        print("TOTAL WORDS = " + str(count+len(self.unk_words)))         
        
        print("Done")
        
        return (all_X_test, all_Y_test)
                                         
        
        
    ##########################  READ UNLABELED DATA  ######################### 
    def read_and_parse_unlabeled_data (self, data_file, skip_unknown_words = False):
        import nltk

        print("Loading unlabeled data from file {}".format(data_file))
        with open(data_file, 'r') as f_data:
            
            self.n_tag_classes = self.DEFAULT_N_CLASSES
            
            all_sentences_words = []
            sentence_words = []
#             raw_tags_test = []        

            # Process all lines in the file
            for line in f_data:
                line = line.strip()                
                
#                 word, tag = line.split('\t') 
                #TODO: replace the tokenizer
                #words = line.split()                 
                
                sentence_words = nltk.word_tokenize(line)              
                
                all_sentences_words.append( tuple(sentence_words) )               
                                    
        print("number of unlabeled examples = " + str(len(all_sentences_words)))
        self.n_sentences_all = len(all_sentences_words)

        # Find the maximum sequence length for Test Data
        #self.max_sentence_len_test = 0
        #for seq in raw_data_test:
        #    if len(seq[0]) > self.max_sentence_len_test:
        #        self.max_sentence_len_test = len(seq[0])
                
#         #Find the maximum sequence length in both training and Testing dataset
#         self.max_sentence_len = max(self.max_sentence_len_train, self.max_sentence_len_test)               
        
        ########################Create UNLABELED DATA Feature Vectors##########################
        all_X_data = []
        word_seq_list = []
        num_tokens_list = []
        self.unk_words = []
        count = 0

        for word_seq in all_sentences_words:  
            if len(word_seq) > self.max_sentence_len_train:
                print("skip the extra words in the long sentence")
                word_seq = word_seq[:self.max_sentence_len_train]

            word_seq_list.append(word_seq)

            elem_wordvecs = []            
            for ix in range(len(word_seq)):
                w = word_seq[ix]
                
                w = w.lower()
                if w in self.word_to_ix_map:
                    count += 1
                    elem_wordvecs.append(self.word_to_ix_map[w])                
                    
                elif "UNK" in self.word_to_ix_map :
                    self.unk_words.append(w)
                    elem_wordvecs.append(self.word_to_ix_map["UNK"])                    
                    
                else:
                    self.unk_words.append(w)
                    w = "UNK"
                    self.word_to_ix_map[w] = self.wordvecs.shape[0] - 1
                    elem_wordvecs.append(self.word_to_ix_map[w])                    
                
            # Pad the sequences for missing entries to make them all the same length
            nil_X = self.zero_vec_pos
            num_tokens_list.append(len(elem_wordvecs))
            pad_length = self.max_sentence_len_train - len(elem_wordvecs)
            all_X_data.append( ((pad_length)*[nil_X]) + elem_wordvecs)            

        all_X_data = np.array(all_X_data)        
        
        print("UNK WORD COUNT = " + str(len(self.unk_words)))
        print("Found WORDS COUNT = " + str(count))
        print("TOTAL WORDS = " + str(count+len(self.unk_words)))         
        
        print("Done")
        
        return (all_X_data, word_seq_list, num_tokens_list)
 