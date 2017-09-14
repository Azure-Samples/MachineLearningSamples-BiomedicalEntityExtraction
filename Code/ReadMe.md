# **Code Structure**
The files in this folder contain the source code for training a word embedding model using Word2Vec on Spark and then to use these embeddings for neural entity recognition. The aim of the project is to be able to
train a domain-specific (Bio Medical) word embedding model and evaluate it against a generic word embedding model. The results we obtain show thata domain-specific word embedding model has superior
performance than the genric one. We use Medline abstracts to train our word embeddings and then use several entity eecognition tasks to evaluate its performance.

Training and evaluating a deep neural network is only the first part of the problem. Operationalizing the trained model in a real-world data processing pipeline, be it real-time or batch scoring, is another challenge. The Operationalization section walks you through the code that will be required to expose the deep learning model as a web service for your external customers. This would also provide a way for deep learning to be used for real-time scoring. To make the life of data scientists easy, we show an end-to-end walkthrough of how to deploy this web service using Docker conatiners on Azure Container Service. We also show how to consume this web service from a website hosted through a docker image on Web App for Linux on Azure.

### [Data Preparation](01_DataPreparation/ReadMe.md)
The goal of this section is to help setup the data for training the word embedding model.

### [Model creation and evaluation](02_Modeling/ReadMe.md)
The goal of this section is to train the word embedding model on Medline Abstracts. WE then show how to use these embeddings as features and train a deep neural network for extracting entities like
"Drugs" and "Diseases" from medical data. We perform several evaluations of the word embedding model as well as the neural entity extractor.

### [Operationalization](03_Deployment/ReadMe.md)
The goal of this section is to operationalize the model created in the previous section and publish a scoring web service. We also demonstrate a basic UI that can used to see how to consume the web service deployed on ACS.
