
# coding: utf-8

# ## Testing the Neural Entity Detector trained using Pubmed Word Embeddings

# ### Step 1

# #### Copy the Embeddings from source location to destination location

# In[ ]:

get_ipython().system(u'cp "Location of Word2Vec_Model.p" .')


# #### Copy the Training Data, Testing Data, Evaluation Script to destination location

# In[1]:

get_ipython().system(u'mkdir Drugs')
get_ipython().system(u'cp "Location of train_drugs.txt" Drugs')
get_ipython().system(u'cp "Location of test_srugs.txt" Drugs')
get_ipython().system(u'cp "Location of evaluation script" Drugs')
get_ipython().system(u'wget https://wcds2017summernlp.blob.core.windows.net/entityrecognition/NERmodel_Drugs.model')
get_ipython().system(u'chmod 777 Drugs/evaldrugs.pl')


# In[1]:

get_ipython().run_cell_magic(u'writefile', u'Data_Preparation.py', u'from keras.preprocessing import sequence\nimport numpy as np\nimport cPickle as cpickle\n\nclass Data_Preparation:\n\n    def __init__ (self, classes, seq_length, train_file=None, test_file=None, vector_size = 100):\n        \n        # Some constants\n        self.DEFAULT_N_CLASSES = classes\n        self.DEFAULT_N_FEATURES = vector_size\n        self.DEFAULT_MAX_SEQ_LENGTH = seq_length\n        \n        # Other stuff\n        self.wordvecs = None\n        self.word_to_ix_map = {}\n        self.n_features = 0\n        self.n_tag_classes = 0\n        self.n_sentences_all = 0\n        self.tag_vector_map = {}\n        \n        self.max_sentence_len_train = 0\n        self.max_sentence_len_test = 0\n        self.max_sentence_len = 0\n        \n        self.all_X_train = []\n        self.all_Y_train = []\n        self.all_X_test = []\n        self.all_Y_test = []\n        self.unk_words = []\n        \n        self.read_and_parse_data(train_file, test_file)\n            \n    def get_data (self):\n        return (self.all_X_train, self.all_Y_train, self.all_X_test, self.all_Y_test, self.wordvecs)\n    \n    def decode_prediction_sequence (self, pred_seq):\n        \n        pred_tags = []\n        for class_prs in pred_seq:\n            class_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)\n            class_vec[np.argmax(class_prs)] = 1\n            if tuple(class_vec.tolist()) in self.tag_vector_map:\n                pred_tags.append(self.tag_vector_map[tuple(class_vec.tolist())])\n            else:\n                print tuple(class_vec.tolist())\n        return pred_tags\n    \n    def read_and_parse_data (self, train_file, test_file, skip_unknown_words = False):\n        \n        ###Load the Word2Vec Model###\n        print("Loading W2V model")\n        W2V_model = cpickle.load(open("Word2Vec_Model.p", "rb"))\n        \n        vocab = list(W2V_model.keys())\n        \n        self.word_to_ix_map = {}\n        self.wordvecs = []\n        \n        ###Create LookUp Table for words and their word vectors###\n        print("Creating LookUp table")\n        for index, word in enumerate(vocab):\n            self.word_to_ix_map[word] = index\n            self.wordvecs.append(W2V_model[vocab[index]])\n        \n        self.wordvecs = np.array(self.wordvecs)\n        print(len(self.wordvecs))\n        self.n_features = len(self.wordvecs[0])\n        print(self.n_features)\n        \n        # Add a zero vector for the Paddings\n        self.wordvecs = np.vstack((self.wordvecs, np.zeros(self.DEFAULT_N_FEATURES)))\n        zero_vec_pos = self.wordvecs.shape[0] - 1\n        \n        ##########################  READ TRAINING DATA  ######################### \n        with open(train_file, \'r\') as f_train:\n            \n            self.n_tag_classes = self.DEFAULT_N_CLASSES\n            self.tag_vector_map = {}    # For storing one hot vector notation for each Tag\n            tag_class_id = 0            # Used to put 1 in the one hot vector notation\n            raw_data_train = []\n            raw_words_train = []\n            raw_tags_train = []        \n\n            # Process all lines in the file\n            for line in f_train:\n                line = line.strip()\n                if not line:\n                    raw_data_train.append( (tuple(raw_words_train), tuple(raw_tags_train)))\n                    raw_words_train = []\n                    raw_tags_train = []\n                    continue\n                \n                word, tag = line.split(\'\\t\')\n                \n                raw_words_train.append(word)\n                raw_tags_train.append(tag)\n                \n                if tag not in self.tag_vector_map:\n                    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)\n                    one_hot_vec[tag_class_id] = 1\n                    self.tag_vector_map[tag] = tuple(one_hot_vec)\n                    self.tag_vector_map[tuple(one_hot_vec)] = tag\n                    tag_class_id += 1\n                    \n        print("raw_nd = " + str(len(raw_data_train)))\n        \n        #Adding a None Tag\n        one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype = np.int32)\n        one_hot_vec[tag_class_id] = 1\n        self.tag_vector_map[\'NONE\'] = tuple(one_hot_vec)\n        self.tag_vector_map[tuple(one_hot_vec)] = \'NONE\'\n        tag_class_id += 1\n        \n        self.n_sentences_all = len(raw_data_train)\n\n        # Find the maximum sequence length for Training data\n        self.max_sentence_len_train = 0\n        for seq in raw_data_train:\n            if len(seq[0]) > self.max_sentence_len_train:\n                self.max_sentence_len_train = len(seq[0])\n                \n                \n        ##########################  READ TEST DATA  ######################### \n        with open(test_file, \'r\') as f_test:\n            \n            self.n_tag_classes = self.DEFAULT_N_CLASSES\n            tag_class_id = 0 \n            raw_data_test = []\n            raw_words_test = []\n            raw_tags_test = []        \n\n            # Process all lines in the file\n            for line in f_test:\n                line = line.strip()\n                if not line:\n                    raw_data_test.append( (tuple(raw_words_test), tuple(raw_tags_test)))\n                    raw_words_test = []\n                    raw_tags_test = []\n                    continue\n                \n                word, tag = line.split(\'\\t\') \n                \n                if tag not in self.tag_vector_map:\n                    print "added"\n                    one_hot_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)\n                    one_hot_vec[tag_class_id] = 1\n                    self.tag_vector_map[tag] = tuple(one_hot_vec)\n                    self.tag_vector_map[tuple(one_hot_vec)] = tag\n                    tag_class_id += 1\n                \n                raw_words_test.append(word)\n                raw_tags_test.append(tag)\n                \n                                    \n        print("raw_nd = " + str(len(raw_data_test)))\n        self.n_sentences_all = len(raw_data_test)\n\n        # Find the maximum sequence length for Test Data\n        self.max_sentence_len_test = 0\n        for seq in raw_data_test:\n            if len(seq[0]) > self.max_sentence_len_test:\n                self.max_sentence_len_test = len(seq[0])\n                \n        #Find the maximum sequence length in both training and Testing dataset\n        self.max_sentence_len = max(self.max_sentence_len_train, self.max_sentence_len_test)\n        \n        ############## Create Train Vectors################\n        self.all_X_train, self.all_Y_train = [], []\n        \n        self.unk_words = []\n        count = 0\n        for word_seq, tag_seq in raw_data_train:  \n            \n            elem_wordvecs, elem_tags = [], []            \n            for ix in range(len(word_seq)):\n                w = word_seq[ix]\n                t = tag_seq[ix]\n                w = w.lower()\n                if w in self.word_to_ix_map :\n                    count += 1\n                    elem_wordvecs.append(self.word_to_ix_map[w])\n                    elem_tags.append(self.tag_vector_map[t])\n\n                elif "UNK" in self.word_to_ix_map :\n                    elem_wordvecs.append(self.word_to_ix_map["UNK"])\n                    elem_tags.append(self.tag_vector_map[t])\n                \n                else:\n                    w = "UNK"       \n                    new_wv = 2 * np.random.randn(self.DEFAULT_N_FEATURES) - 1 # sample from normal distribution\n                    norm_const = np.linalg.norm(new_wv)\n                    new_wv /= norm_const\n                    self.wordvecs = np.vstack((self.wordvecs, new_wv))\n                    self.word_to_ix_map[w] = self.wordvecs.shape[0] - 1\n                    elem_wordvecs.append(self.word_to_ix_map[w])\n                    elem_tags.append(list(self.tag_vector_map[t]))\n\n            \n            # Pad the sequences for missing entries to make them all the same length\n            nil_X = zero_vec_pos\n            nil_Y = np.array(self.tag_vector_map[\'NONE\'])\n            pad_length = self.max_sentence_len - len(elem_wordvecs)\n            self.all_X_train.append( ((pad_length)*[nil_X]) + elem_wordvecs)\n            self.all_Y_train.append( ((pad_length)*[nil_Y]) + elem_tags)\n\n        self.all_X_train = np.array(self.all_X_train)\n        self.all_Y_train = np.array(self.all_Y_train)\n        \n        ########################Create TEST Vectors##########################\n\n        self.all_X_test, self.all_Y_test = [], []\n        \n        for word_seq, tag_seq in raw_data_test:  \n            \n            elem_wordvecs, elem_tags = [], []            \n            for ix in range(len(word_seq)):\n                w = word_seq[ix]\n                t = tag_seq[ix]\n                w = w.lower()\n                if w in self.word_to_ix_map:\n                    count += 1\n                    elem_wordvecs.append(self.word_to_ix_map[w])\n                    elem_tags.append(self.tag_vector_map[t])\n                    \n                elif "UNK" in self.word_to_ix_map :\n                    self.unk_words.append(w)\n                    elem_wordvecs.append(self.word_to_ix_map["UNK"])\n                    elem_tags.append(self.tag_vector_map[t])\n                    \n                else:\n                    self.unk_words.append(w)\n                    w = "UNK"\n                    self.word_to_ix_map[w] = self.wordvecs.shape[0] - 1\n                    elem_wordvecs.append(self.word_to_ix_map[w])\n                    elem_tags.append(self.tag_vector_map[t])\n                \n            # Pad the sequences for missing entries to make them all the same length\n            nil_X = zero_vec_pos\n            nil_Y = np.array(self.tag_vector_map[\'NONE\'])\n            pad_length = self.max_sentence_len - len(elem_wordvecs)\n            self.all_X_test.append( ((pad_length)*[nil_X]) + elem_wordvecs)\n            self.all_Y_test.append( ((pad_length)*[nil_Y]) + elem_tags)\n\n        self.all_X_test = np.array(self.all_X_test)\n        self.all_Y_test = np.array(self.all_Y_test)\n        \n        print("UNK WORD COUNT " + str(len(self.unk_words)))\n        print("Found WORDS COUNT " + str(count))\n        print("TOTAL WORDS " + str(count+len(self.unk_words)))\n        \n        return (self.all_X_train, self.all_Y_train, self.all_X_test, self.all_Y_test, self.wordvecs)\n ')


# In[7]:

from Data_Preparation import Data_Preparation
from NER_Model import NER_Model
import cPickle as cp
from keras.models import load_model
import numpy as np

TRAIN_FILEPATH = "Drugs//train_drugs.txt"
TEST_FILEPATH = "Drugs//test_drugs.txt"

vector_size = 50
classes = 7 + 1
seq_length = 613
layer_arg = 2
ep_arg = 10

if __name__ == "__main__":

    reader = Data_Preparation(classes, seq_length, TRAIN_FILEPATH, TEST_FILEPATH, vector_size)
    
    X_train, Y_train, X_test, Y_test, wordvecs = reader.get_data()

    nermodel = load_model("NERmodel_Drugs.model")
    
    # Evaluate the model
    print("Evaluating model...")
    target = open("Pubmed_Output.txt", 'w')
    predicted_tags= []
    test_data_tags = []
    ind = 0
    for x,y in zip(X_test, Y_test):
        tags = nermodel.predict(np.array([x]), batch_size=1)[0]
        pred_tags = reader.decode_prediction_sequence(tags)
        test_tags = reader.decode_prediction_sequence(y)
        ind += 1
        ### To see Progress ###
        if ind%500 == 0: 
            print("Sentence" + str(ind))

        pred_tag_wo_none = []
        test_tags_wo_none = []

        for index, test_tag in enumerate(test_tags):
            if test_tag != "NONE":
                test_tags_wo_none.append(test_tag)
                pred_tag_wo_none.append(pred_tags[index])

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
    
    print("Done.") 


# In[8]:

file1 = open("Pubmed_Output.txt")
file2 = open("Drugs//test_drugs.txt")
target = open("Drugs//eval2.txt", "w")

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


# In[9]:

get_ipython().system(u'./Drugs/evaldrugs.pl Drugs/eval2.txt Drugs/test_drugs.txt')

