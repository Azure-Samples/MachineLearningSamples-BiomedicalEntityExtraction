# **Medical Entity Extraction using Deep Learning**

## **Problem definition**
Medical Named Entity Recognition is a critical step for complex biomedical tasks such as understanding the interactions between different entity types like Drugs, Diseases etc. This study focuses on how a large amount of unstructured data available form Medline Abstracts can be utilized for training a Neural Entity Extractor. We systematically show how to train a Word Embeddings Model using nearly 15 million Medline abstracts using Word2Vec on Spark and then use them to build a LSTM based deep neural network for Entity Extraction on a GPU enabled VM on Azure. Our results show that training a domain specific word embedding model boosts the performance when compared to embeddings trained on Generic data such as Google News. The in-domain word embedding model can detect 7012 entities correctly (out of 9475) with a F1 score of 0.73 compared to 5274 entities with F1 score of 0.61 for generic word embedding model.

We also demonstrate how we can publish the trained Neural Network as a service for real time scoring using Docker and Azure Container Service. Finally, we develop a basic website using Flask to consume the created web service and host it on Azure using Web App for Linux. Currently the model operational on the website (http://medicalentitydetector.azurewebsites.net/) supports 7 entity types namely, Diseases, Drug or Chemicals, Proteins, DNA, RNA, Cell Line, Cell Type.

### What this repository provides
This repository provides the notebooks that can be used to train a Word2Vec Model on spark followed by training a Neural Entity Extractor followed by deploying the trained deep neural network. We have included the formatted datasets used with each notebook in the relevant folders. While trying to replicate the results the user can mention the detailed path of these datasets and can run the notebooks. 

The code has been structured in the following format 

* Data Preparation: 
    * [Downloading and Parsing the Medline Abstracts](Code/01_DataPreparation/ReadMe.md)

* Modeling
    * Feature Engineering
        * [Training Word2Vec Model](Code/02_Modeling/01_FeatureEngineering/ReadMe.md)

    * Model Creation
        * [Building Neural Entity Extractor](Code/02_Modeling/02_ModelCreation/ReadMe.md)

    * Model Evaluation
        * [Evaluate the NER Model](Code/02_Modeling/03_ModelEvaluation/ReadMe.md)

* Operationalization
    * [Operationalize the trained model](Code/03_Deployment/ReadMe.md)

### Tools required for replicating the pipeline

* [Spark HDInsight Cluster](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) version Spark 2.1 on Linux (HDI 3.6) and associated container
* [NC6 Data Science VM](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro)
* [Tensorflow](https://www.tensorflow.org/install/)
* [CNTK 2.0](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras)
* [Keras](https://keras.io/#installation)
* [Anaconda with Python 2.7](https://www.continuum.io/downloads)
* [Jupyter Notebook](http://jupyter.readthedocs.io/en/latest/install.html)



![Architecture](../../../Images/Architecture.png)



## **[Data Acquisition and Understanding](Code/01_DataPreparation/ReadMe.md)**
The first step in our pipeline is obtain the data from [MEDLINE](https://www.nlm.nih.gov/pubs/factsheets/medline.html). The data is available publically and is in form of XML files available on their [FTP server](ftp://ftp.nlm.nih.gov/nlmdata/.medleasebaseline/gz/). There are 812 XML files available on the server and each of the XML files contain around 30000 abstracts. The fields present in each file are 
        
        abstract
        affiliation
        authors
        country	
        delete: boolean if False means paper got updated so you might have two XMLs for the same paper.
        file_name	
        issn_linking	
        journal	
        keywords	
        medline_ta: this is abbreviation of the journal nam	
        mesh_terms: list of MeSH terms	
        nlm_unique_id	
        other_id: Other IDs	
        pmc: Pubmed Central ID	
        pmid: Pubmed ID
        pubdate: Publication date
        title

This amount to a total of 24 million abstracts but nearly 10 million documents do not have a field for abstracts. Since the amount of data to be processed is huge and cannot be loaded into memory at a single instance
we rely on Sparks Distributed Computing capabilities for processing. Once the data is available in Spark as a data frame we can apply other pre-processing techniques on it like training the Word Embedding Model. Refer [this](Code/01_DataPreparation/ReadMe.md) to get started.


Data after parsing XMLs

![Data Sample](../../../Images/datasample.png)

Other datasets which are being used for training and evaluation of the Neural Entity Extractor have been include in the corresponding folder. To obtain more information about them you could refer to the following corpora.
 * [Bio-Entity Recognition Task at Bio NLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html)
 * [BioCreative V CDR task corpus](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
 * [Semeval 2013 - Task 9.1 (Drug Recognition)](https://www.cs.york.ac.uk/semeval-2013/task9/)


## **[Modeling](Code/02_Modeling/ReadMe.md)**
This is the stage where we show how you can use the data downloaded in the previous section for training your own word embedding model and use it for other downstream tasks. Although the we are using the Medline data, however the pipeline to generate the embeddings is generic and can be reused to train word embeddings for any other domain. For embeddings to be an accurate representation of the data it is essential that the word2vec is trained on a large amount of data. 

Once we have the word embeddings ready we can make a deep neural network which uses the learnt embeddings to initialize the Embedding layer. We mark the embedding layer as non-trainable but that is not mandatory. The training of the word embedding model is totally unsupervised and hence we are able to take advantage of unstructured texts. However, to train an entity recognition model we need labeled data. The more the better.

### **[Training Word2Vec](Code/02_Modeling/01_FeatureEngineering/ReadMe.md)**
Word2Vec is the name given to a class of neural network models that, given an unlabeled training corpus, produce a vector for each word in the corpus that encodes its semantic information. These models are simple neural networks with one hidden layer. The word vectors/embeddings are learned by backpropagation and stochastic gradient descent. There are 2 types of word2vec models, namely, the Skip-Gram and the continuous-bag-of-words. Since we are using the [MLlib's](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec) implementation of the word2vec which supports the Skip-gram model we will briefly describe the model here. For details see [this](https://arxiv.org/pdf/1301.3781.pdf).

![Skip Gram Model](../../../Images/Skip Gram.png)

The model uses Hierarchical Softmax and Negative sampling to optimize the performance. Hierarchical SoftMax (H-SoftMax) is an approximation inspired by binary trees. H-SoftMax essentially replaces the flat SoftMax layer with a hierarchical layer that has the words as leaves. This allows us to decompose calculating the probability of one word into a sequence of probability calculations, which saves us from 
having to calculate the expensive normalization over all words. Since a balanced binary tree has a depth of log2(|V|)log2(|V|) (V is the Vocabulary), we only need to evaluate at most log2(|V|)log2(|V|) nodes to obtain the final probability of a word. The probability of a word w given its context c is then simply the product of the probabilities of taking right and left turns respectively that lead to its leaf node. We can build a Huffman Tree based on the frequency of the words in the dataset to ensure that more frequent words get shorter representations. Refer [this](http://sebastianruder.com/word-embeddings-softmax/) for further information.
Image taken from [here](https://ahmedhanibrahim.wordpress.com/2017/04/25/thesis-tutorials-i-understanding-word2vec-for-word-embedding-i/)

Once we have the embeddings we would like to visualize them and see the relationship between semantically similar words. 

![W2V similarity](../../../Images/W2v_sim.png)

We have shown 2 different ways of visualizing the embeddings. The first, uses a PCA to project the high dimensional vector to a 2-D vector space. This leads to a significant loss of information and the visualization is not as accurate. The second is to use PCA with t-SNE. t-SNE is a nonlinear dimensionality reduction technique that is particularly well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot.  It models each high-dimensional object by a two- or three-dimensional point in such a way that similar objects are modeled by nearby points and dissimilar objects are modeled by distant points. It works in 2 parts. First, it creates a probability distribution over the pairs in the higher dimensional space in a way that similar objects have a high probability of being picked and dissimilar points have  very low probability of getting picked. Second, it defines a similar probability distribution over the points in a low dimensional map and minimizes the KL Divergence between the two distributions with respect to location of points on the map. The location of the points in the low dimension is obtained by minimizing the KL Divergence using Gradient Descent. But t-SNE might not be always reliable. Refer [this](https://distill.pub/2016/misread-tsne/) Refer [this](Code/02_Modeling/01_FeatureEngineering/ReadMe.md) for details about the implementation.

### Visualization with PCA
![PCA](../../../Images/pca.png)

### Visualization with t-SNE
![t-SNE](../../../Images/tsne.png)

### Points closest to Cancer (they are all types of Cancer)
![Points closest to Cancer](../../../Images/nearesttocancer.png)

### **[Training the Neural Entity Extractor](Code/02_Modeling/02_ModelCreation/ReadMe.md)**
Traditional Neural Network Models suffer from a problem that they treat each input and output as independent of the other inputs and outputs. This may not be a good idea for tasks such as Machine translation, Entity Extraction or any other sequence to sequence labelling tasks. Recurrent Neural Network models overcome this problem as they can pass information computed till now to the next node. This property is called having memory in the network since it is able to use the previously computed information. The below picture represents this.

![RNN](../../../Images/RNN_expanded.png)

Vanilla RNNs actually suffer from the [Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) due to which they are not able to utilize all the information they have seen before. This problem becomes evident only when a large amount of context is required to make a prediction. But models like LSTM do not suffer from this problem, in fact they are designed to remember long term dependencies. Unlike vanilla RNNs that have a single neural network, the LSTMs have the interactions between 4 neural networks for each cell. Refer this [excellent post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a detailed explanation of how LSTM work.

![LSTM Cell](../../../Images/lstm cell.png)

Equipped with this information, let’s try and put together our own LSTM based Recurrent Neural Network and try to extract Entities like Drugs, Diseases etc. from Medical Data. The first step is to obtain a large amount of labelled data and as you would have guessed, that's not easy! Most of the medical data contains lot of sensitive information about the person and hence are not publicly available. We rely on a combination of 2 different datasets that are publicly available. The first dataset is from Semeval 2013 - Task 9.1 (Drug Recognition) and the other is from BioCreative V CDR task. We are combining and auto labelling these 2 datasets so that we can detect both drugs and diseases from medical texts and evaluate our word embeddings. 

Refer [this](Code/02_Modeling/02_ModelCreation/ReadMe.md) for implementation details

The model architecture that we have used across all the codes and for comparison is presented below. The parameter that changes for different daatsets is the maximum sequence length (613 here).

![LSTM model](../../../Images/D_a_D_model.png)

### **Results**

We use the evaluation script from the shared task [Bio-Entity Recognition Task at Bio NLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) to evaluate the precision, recall and F1 score of the model. Below is the comparison of the results we get with the embeddings trained on Medline Abstracts and that on Google News embeddings. We clearly see that the in-domain model is out performing the generic model. Hence having a specific word embedding model rather than using a generic one is much more helpful. 

![Model Comparison 1](../../../Images/mc1.png)

We perform the evaluation of the word embeddings on other datasets in the similar fashion and see that in-domain model is always better.

![Model Comparison 2](../../../Images/mc2.png)

![Model Comparison 3](../../../Images/mc3.png)

![Model Comparison 4](../../../Images/mc4.png)

All the training and evaluations reported here are done using Keras and TensorFlow. Keras also supports CNTK backend but since it does not have all the functionalities for the bidirectional model yet we have used unidirectional model with CNTK backend to benchmark the results of CNTK model with that of TensorFlow. These are the results we get

![Model Comparison 5](../../../Images/mc5.png)

We also compare the performance of Tensorflow vs CNTK and see that CNTK performs as good as Tensorflow both in terms of time taken per epoch (60 secs for CNTK and 75 secs for Tensorflow) and the number of entities detected. We are using the Unidirectional layers for this evaluation.

![Model Comparison 6](../../../Images/mc6.png)


## **[Operationalization](Code/03_Deployment/ReadMe.md)**
Training a Neural Network and getting good results is the first part of the problem. Next we want to be able to expose our model as a web service which people can consume directly. 
But the problem with this is that usually a deep learning model has a lot of dependencies. Making sure that all the dependencies are installed in the machines which you use for deployment is often a challenging task. So what can we do about it? One of the ways to go about it is to use [Docker Containers](https://blogs.msdn.microsoft.com/uk_faculty_connection/2016/09/23/getting-started-with-docker-and-container-services/). Once we have our containers ready we can take those containers to Azure and deploy it on Azure Container Service. This helps to do real time scoring for test inputs coming through the web service. We use this [tutorial](https://gallery.cortanaintelligence.com/Tutorial/Deploy-CNTK-model-to-ACS) to setup the operationalization of our deep learning model. 

![Website](../../../Images/website.png)

### **Conclusion** 
In this report, we went over the details of how you could train a Word Embedding Model using Word2Vec on Spark and then use the Embeddings obtained for training a Neural Network for Entity Extraction. We have shown the pipeline for Bio-Medical domain but the pipeline is generic. You just need enough data and you can easily adapt the workflow presented here for a different domain.

