## **Code Structure for Model Creation and Evaluation**

### [01_FeatureEngineering](01_FeatureEngineering/ReadMe.md)
The notebook in the feature engineering section details how the Word2Vec model can be used to extract word embeddings for Bio-Medical data. We are using the Medline abstracts downloaded and parsed in the Data Preparation 
section. These word embeddings then act as features for our Neural Entity Extraction model. The notebook also shows how to evaluate the quality of the embeddings using intrinsic evaluation and goes on to demonstrate methods for visualization of the embeddings using PCA and t-SNE. The computing resource used here is Spark.

### [02_ModelCreation](02_ModelCreation/ReadMe.md)
The notebook in this section covers the details of how we can use the word embeddings obtained in the previous section to initialize the embedding layer of our neural network. It provides a mechanism of how you can
convert the input data (which is in BIO format) to an input shape which Keras understands. Once you convert the data in this format the notebook details the steps of how you can create a deep neural network by 
adding different layers. Towards the end, the notebook describes how to evaluate the performance of the model on the test set using an evaluation script.

### [03_ModelEvaluation](03_ModelEvaluation/ReadMe.md)
This section is about comparing the model to different baseline models and evaluating how the domain-specific word embeddings perform when compared with generic embeddings. We are using the embeddings trained on 
Google News to perform the comparisons. We find that training domain-specific entities gives a significant improvement over the generic word embeddings. In this section we also compare the performance of the 
CNTK backend and the Tensorflow Backend. We find that CNTK is faster than Tensorflow but since the CNTK backend [does not](https://docs.microsoft.com/en-us/cognitive-toolkit/Using-CNTK-with-Keras#known-issues) 
have a complete implementation of BiDirectional LSTM Layer the performance is not as good.
