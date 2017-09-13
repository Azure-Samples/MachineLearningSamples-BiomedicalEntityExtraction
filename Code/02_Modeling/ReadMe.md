##**Code Structure for Model Creation and Evaluation**

### [01_FeatureEngineering](01_FeatureEngineering/ReadMe.md)
The Notebook in the feature engineering section details how the Word2Vec Model can be used to extract Word Embeddings for Bio-Medical Data. We are using the Medline Abstracts downloaded and parsed in the Data Preparation 
section. These word embeddings then act as features for our Neural Entity Extraction Model. The Notebook also shows how to evaluate the quality of the embeddings using intrinsic evaluation and goes on to 
demonstrate methods for visualization of the embeddings using PCA and t-SNE. The computing resource used here is Spark.

### [02_ModelCreation](02_ModelCreation/ReadMe.md)
The Notebook in this section covers the details of how we can use the word embeddings obtained in the previous section to initialize the Embedding Layer of our Neural Network. It provides a mechanism of how you can
convert the input data (which is in BIO format) to an input shape which Keras understands. Once you convert the data in this format the notebook details the steps of how you can create a deep neural network by 
adding different Layers. Towards the end, the notebook describes how to evaluate the performance of the model on the test set using an Evaluation script.

### [03_ModelEvaluation](03_ModelEvaluation/ReadMe.md)
This section is about comparing the model to different baseline modls and evaluate how the domain specific word embeddings perform when compared with Generic embeddings. We are using the Embeddings trained on 
Google News to perform the comparisons. We find that training domain specific entities gives a significant improvement over the Generic Word Embeddings. In this section we also compare the performance of the 
CNTK backend and the Tensorflow Backend. We find that CNTK is faster than Tensorflow but since the CNTK backend [does not](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras#known-issues) 
have a complete implementation of BiDirectional LSTM Layer the performance is not as good.
