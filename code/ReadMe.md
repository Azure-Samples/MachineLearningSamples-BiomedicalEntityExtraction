# **Code Structure**
The files in this folder contain the source code for training a word embedding model using Word2Vec on Spark and then to use these embeddings for Neural Entity Recognition. The aim of the project is to be able to
train a domain specific (Bio Medical Domain) word embedding model and evaluate it against Generic word Embedding Model. The evaluations we perform justify that having a Domain Specific Word Embedding model has superior
performance than the Genric one. We are using Medline abstracts to train our Word Embeddings and then using several Entity Recognition Tasks to evaluate its performance. 

Training and Evaluating a Deep Neural Network is only the first part of the problem. Operationalizing the trained model in a real-world data processing pipeline, be it real-time or batch scoring is another 
challenge. The Operationalization section walks you through the code that will be required to expose the deep learning model as a web service for external customers of yours. This would also provide a way how deep learning can be used for real time scoring. 

### [Data Preparation](01_data_acquisition_and_understanding/ReadMe.md)
The goal of this section is to help setup the data for training the word embedding model. 

### [Model creation and evaluation](02_modeling/ReadMe.md)
The goal of this section is to train the word embedding model on Medline Abstracts. Then to use these embeddings as features and train a deep neural network for extracting entities like
Drugs, Diseases from medical data. We perform several evaluations of the word embedding model as well as the neural entity extractor. 

### [Deployment](03_deployment/ReadMe.md)
The goal of this section is to operationalize the model created in the previous section and publish a scoring web service. We also demonstrate a basic UI that can used to see how to 
consume the web service deployed on ACS. 
