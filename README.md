# **Code Structure**
The files in this folder contain the source code for training a word embedding model using Word2Vec on Spark and then to use these embeddings for Neural Entity Recognition. The aim of the project is to be able to train a domain specific (Biomedical domain) word embedding model and evaluate it against generic word embedding model. The evaluation results have shown that a domain-specific word embedding model outperfoms the generic one when used as word features for entity extraction. We are using Medline PubMed article abstracts to extract word embeddings and then using that as features to train neural models to detect different entity types in the biomedical domain. The code is generic to be applied to train an entity extractio model on any other domain given big amount of unlabeled data to train the word embedding model and a fair amount of labeled data to train the LSTM neural network model.

In real-world scenarios, training and evaluation the quality of a deep neural network is the first part of the solution. Operationalizing the trained model in a data processing pipeline either for real-time or batch scoring is one of the key capabilities of Azure Machine Learning service. The Operationalization section walks you through the required code to expose the deep learning model as a web service for your model consumers. This section shows how a deep learning model could be used for real time scoring. 

### [Data Acquisition and Understanding](01_data_acquisition_and_understanding/ReadMe.md)
The goal of this section is to help setup the data for training the word embedding model. 

### [Modeling](02_modeling/ReadMe.md)
The goal of this section is to train the word embedding model on Medline Abstracts. Then to use these embeddings as features and train a deep neural network for extracting entities like
Drugs, Diseases from medical data. We perform several evaluations of the word embedding model as well as the neural entity extractor. 

### [Deployment](03_deployment/ReadMe.md)
The goal of this section is to operationalize the model created in the previous section and publish a scoring web service. We also demonstrate a basic UI that can used to see how to 
consume the web service deployed on ACS. 
