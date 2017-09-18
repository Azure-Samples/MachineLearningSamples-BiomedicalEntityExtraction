### 2. [Modeling](./code/02_modeling)

Modeling is the stage where we show how you can use the data downloaded in the previous section for training your own word embedding model and use it for other downstream tasks. Although we are using the PubMed data, the pipeline to generate the embeddings is generic and can be reused to train word embeddings for any other domain. For embeddings to be an accurate representation of the data, it is essential that the word2vec is trained on a large amount of data.
Once we have the word embeddings ready, we can train a deep neural network model that uses the learned embeddings to initialize the Embedding layer. We mark the embedding layer as non-trainable but that is not mandatory. The training of the word embedding model is unsupervised and hence we are able to take advantage of unlabeled texts. However, the training of the entity recognition model is a supervised learnign task and its accuracy depends on the amount and teh quality of a manually-annotated data. 

### 2.1 [Feature Engineering](01_feature_engineering/ReadMe.md)
The Python script in this step runs on a Spark cluster. It shows how to train a Word2Vec model on a big data corpus. We are using the Medline Abstracts downloaded and parsed in the [Data Acquisition and Understanding](./code/01_Data_Acquisition_and_Understanding/ReadMe.md) 
section. These word embeddings then act as features for our Neural Entity Extraction Model. The computing resource used here is Spark.

### 2.2 [Model Creation](02_model_creation/ReadMe.md)
The Python script in this step covers the details of how we can use the word embeddings obtained in the previous section to initialize the embedding layer of the LSTM neural network. It provides a mechanism of how you can
convert the input data (which is in BIO format) to an input shape which Keras understands. Once you convert the data in this format the notebook details the steps of how you can create a deep neural network by 
adding different Layers. Towards the end, the notebook describes how to evaluate the performance of the model on the test set using an Evaluation script.

### 2.3 [Model Evaluation](03_model_evaluation/ReadMe.md)
This section is about comparing the model to different baseline modls and evaluate how the domain specific word embeddings perform when compared with Generic embeddings. We are using the Embeddings trained on 
Google News to perform the comparisons. We find that training domain specific entities gives a significant improvement over the Generic Word Embeddings. In this section we also compare the performance of the 
CNTK backend and the Tensorflow Backend. We find that CNTK is faster than Tensorflow but since the CNTK backend [does not](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras#known-issues) have a complete implementation of BiDirectional LSTM Layer the performance is not as good.
