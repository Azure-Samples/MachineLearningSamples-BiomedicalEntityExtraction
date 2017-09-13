## [Training a Neural Entity Detector using Pubmed Word Embeddings](3_Training_Neural_Entity_Extractor_Pubmed.ipynb)
This [Notebook](3_Training_Neural_Entity_Extractor_Pubmed.ipynb) describes how you can use [Keras](https://keras.io/) with [Tensorflow](https://www.tensorflow.org/) backend to train a Deep Neural Network for Entity Recognition. We demonstrate how we can use the Word Embeddings generated previously to initialize the Embedding layer
of the Deep Neural Network. The task at hand is to identity Drugs and Diseases from a given text. We are using an auto labeled dataset which is the combination of Semeval 2013 - Task 9.1 (Drug Recognition) and BioCreative V CDR task corpus.


This Notebook uses a [Linux Data Science VM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) by Azure which has a single GPU, its [NC6](https://azure.microsoft.com/en-us/pricing/details/virtual-machines/series/#n-series).
Before, proceeding forward make sure you have the word embeddings model trained. You can refer to this [notebook](../01_FeatureEngineering/2_Train_Word2Vec.ipynb)
to see how to train your own word embedding model for the Bio-Medical domain using Word2Vec on Spark.

Once you have the Embedding model ready you can start working on training the Neural Network for Entity Recognition.

**Step 1**: Copy the dataset and evaluation scripts to correct locations and Read Word Embeddings from Parquet Files:
We are using the [fastparquet package](https://pypi.python.org/pypi/fastparquet) as a way to read the embeddings from the parquet files and load them to a Pandas dataframe. You can then store this embedding matrix 
in any format and use it either as a TSV to visualize using the [Projector for Tensorflow](http://projector.tensorflow.org/) or use as a lookup table for downstream Deep Learning Tasks. The code provides a way to download the files from the Blob connected to your Spark Cluster.
for more information about blob storage see [this](https://docs.microsoft.com/en-us/azure/storage/storage-dotnet-how-to-use-blobs)

**Step 2**: Prepare the data for training and testing in a format that is suitable for Keras. The read_and_parse_data function is the one that does that.
 - It first reads the word embeddings and a word_to_index_map mapping each word in the embeddings to an index. It also creates a list where each item id refers to the the word vector corresponding to that index and hence to its word.
 - Next it reads the training and testing data and line by line and appends a sentence to a list. It also creates one-hot vectors for each of the classes of Tags (like B-Disease, I-Disease, B-Drug etc.)
 - Once the list of sentences is ready, its time now to replace each word with its index from the above map. If we find a word which is not present in our vocabulary we replace the word by the token "UNK".
 To generate the vector for "UNK" we sample a random vector, which has the same dimension as our embeddings, from a Normal Distribution. Since the number of words in each sentence might differ, we pad each sequence 
 to make sure that they have the same length. We add an additional tag "NONE" for each of the padded term. We also associate a zero vector with the paddings. The final shape of the train and test data should be 
 (number of samples, max_sequence_length). This is the shape that can be fed to the [Embedding Layer](https://keras.io/layers/embeddings/) in Keras. Once we have this shape for our dataset we are ready for training 
 our Neural Network (but first lets create one).
 
 
 **Step 3**: This step decribes how to create a Deep Neural Network in Keras. We first create a [sequential](https://keras.io/getting-started/sequential-model-guide/) model for our Neural Network.
 We start by adding an [Embedding Layer](https://keras.io/layers/embeddings/) to our model and specify the input shape as created above. We load our pre-trained Embeddings for the weights of this layer and set the *trainable* flag as False since we do not want 
 to update the Embeddings (but this can change). Next we add a [Bi-Directional LSTM layer](https://keras.io/layers/wrappers/#bidirectional). We add a [dropout layer](https://keras.io/layers/core/#dropout). 
 We repeat the previous step once again. Finaly, we add a [TimeDistributed Dense Layer](https://keras.io/layers/wrappers/#timedistributed). This layer is responsible for generating predictions for each word in the sentence.
 Our model looks like this
        
        Embedding Layer
        Bidirectional LSTM Layer    
        Dropout Layer
        BiDirectional LSTM Layer
        Dropout Layer
        TimeDistributed Dense Layer 

![LSTM model](../../../Images/D_a_D_model.png)

We optimize the [categorical_crossentropy](https://keras.io/losses/#categorical_crossentropy) loss and are using the [Adam](https://keras.io/optimizers/#adam) optimizer.

**Step 4**: Now we have the data ready and the Neural Network Built, so lets put them together and start the training. This step shows how to call the previously 
defined functions. We specify the paths of the training and the test files along with some parameters like 

        vector size: this is the length of a word vector (50 for us).
        classes:     this is the number of unique classes in the training and test sets (eg of a class would be B-Chemical, I-Disease, O etc.)
        seq_length:  this is the max_sequence_length found above
        layers:      number of LSTM layers you want to have
        epochs:      number of epochs you would like to do for Neural Network Training

Once these are set, the model we start to train. Next step would be to obtain model predictions on the test set and evaluate the performance of the model.

**Step 5**: We store the predictions obtained from the previous step into a text file. The first step here will to combine that output and obtain a file in the 
following format

        Word1   Tag1
        Word2   Tag2
        Word3   Tag3

        Word4   Tag4
        Word5   Tag5

Once we have the output in the above format we can use the SharedTaskEvaluation Script to obtain the recall, precision and F1-score for our Model.

![Sample Evaluation](../../../Images/Evaluation_Sample.png)




