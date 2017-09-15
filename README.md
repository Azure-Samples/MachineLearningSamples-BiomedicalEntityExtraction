# Biomedical Entity Recognition from MEDLINE Abstracts using Deep Learning

Link to the public GitHub repository:

[https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction](https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction)


## Detailed documentation in GitHub repository
We provide summary documentation here about the sample. More extensive documentation can be found on the GitHub site in the file: [https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/ProjectReport.md](https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/ProjectReport.md).


[quick-start-installation.md] **AT** MISSING LINK

## Prerequisites

### Azure subscription and Hardware
1. An Azure [subscription](https://azure.microsoft.com/en-us/free/)
2. [HDInsight Spark Cluster](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) version Spark 2.1 on Linux (HDI 3.6). To process the full amount of MEDLINE abstracts discussed below, you need the minimum configuration of:
    - Head node: [D13_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/) size
    - Worker nodes: At least 4 of [D12_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/). In our work, we used 11 worker nodes of D12_V2 size.

3. [NC6 Data Science Virtual Machine (DSVM)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) on Azure.

### Software
1. Azure Machine Learning Workbench. See [installation guide](quick-start-installation.md). LINK DIDN'T WORK FOR ME
2. [TensorFlow](https://www.tensorflow.org/install/)
3. [CNTK 2.0](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras)
4. [Keras](https://keras.io/#installation)


## Introduction: Business Case Understanding & Project Summary

### Use Case Overview
Medical Named Entity Recognition (NER) is a critical step for complex biomedical Natural Language Processing (NLP) tasks such as: 
* Extraction of diseases and symptoms from electronic medical or health records.
* Understanding the interactions between different entity types like drugs and diseases for the purpose of pharmacovigilence.

Our study focuses on how a large amount of unstructured biomedical data available from MEDLINE abstracts can be utilized for training a Neural Entity Extractor for biomedical NER.

**AT** PERHAPS HERE YOU COULD GIVE AN EXAMPLE OF A SENTENCE FROM THE BIOMEDICAL DOMAIN AND "TAG" IT FOR NAMED-ENTITIES 

 The project highlights several features of Azure Machine Learning Workbench, such as:
 1. Instantiation of the [Team Data Science Process (TDSP) structure and templates](how-to-use-tdsp-in-azure-ml.md )
 2. Execution of code in Jupyter notebooks as well as Python files
 3. Run history tracking for Python files
 4. Execution of jobs on remote Spark compute context using HDInsight Spark 2.1 clusters
 5. Execution of jobs in remote GPU VMs on Azure
 6. Easy operationalization of Deep Learning models as web services on Azure Container Services

### Purpose of the Project
The objectives of the sample are: 
1. Show how to systematically train a word embeddings model using nearly 15 million MEDLINE abstracts using [Word2Vec on Spark](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec). These word embeddings are then used to build an LSTM-based deep neural network for Entity Extraction on a GPU enabled VM on Azure.
2. Demonstrate that domain-specific data can enhance accuracy of NER compared to generic data, such as Google News, which are often used for such tasks.
3. Demonstrate an end-to-end workflow of how to train and operationalize deep learning models on large amounts of text data using Azure Machine Learning Workbench and multiple compute contexts (Spark, GPU VMs).
4. Demonstrate the capabilities of Azure Machine Learning Workbench **AT** REMOVED REPETITION

### Summary: Results and Deployment
Our results show that training a domain-specific word embedding model boosts the accuracy of biomedical NER when compared to embeddings trained on generic data such as Google News. The in-domain word embedding model can detect 7012 entities correctly (out of 9475) with a F1 score of 0.73 compared to 5274 entities with F1 score of 0.61 for generic word embedding model.

We also demonstrate how we can publish the trained neural network as a web service for real time scoring using Docker and Azure Container Service. Finally, we develop a basic website using Flask to consume the created web service and host it on Azure using Web App for Linux. Currently the model operational on the website (http://medicalentitydetector.azurewebsites.net/) supports seven entity types, namely, "Diseases", "Drug or Chemicals", "Proteins", "DNA", "RNA", "Cell Line" and "Cell Type".


## Architecture
The figure shows the architecture that was used to process data and train models.

![Architecture](./Images/architecture.png)

## Data 
We first obtained the raw MEDLINE abstract data from [MEDLINE](https://www.nlm.nih.gov/pubs/factsheets/medline.html). The data is available publically and is in the form of XML files available on their [FTP server](ftp://ftp.nlm.nih.gov/nlmdata/.medleasebaseline/gz/). There are 812 XML files available on the server and each of the XML files contain around 30 million abstracts. More detail about data acquisition and understanding is provided in the Project Structure. The fields present in each file are 
        
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

This amount to a total of 24 million abstracts but nearly 10 million documents do not have a field for abstracts. Since the amount of data processed is large and cannot be loaded into the memory of a single machine, we rely on HDInsight Spark for processing. Once the data is available in Spark as a data frame, we can apply other pre-processing techniques on it like training the Word Embedding Model. Refer to [./Code/01_DataPreparation/ReadMe.md](./Code/01_DataPreparation/ReadMe.md) to get started.


Data after parsing XMLs:

![Data Sample](./Images/datasample.png)

Other datasets, which are being used for training and evaluation of the Neural Entity Extractor have been include in the corresponding folder. To obtain more information about them, you could refer to the following corpora:
 * [Bio-Entity Recognition Task at Bio NLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html)
 * [BioCreative V CDR task corpus](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
 * [Semeval 2013 - Task 9.1 (Drug Recognition)](https://www.cs.york.ac.uk/semeval-2013/task9/)


## Project Structure and Reporting According TDSP LifeCycle Stages
For the project, we use the TDSP folder structure and documentation templates (Figure 1), which follows the [TDSP lifecycle](https://github.com/Azure/Microsoft-TDSP/blob/master/Docs/lifecycle-detail.md). The project is created based on instructions provided [here](https://github.com/amlsamples/tdsp/blob/master/Docs/how-to-use-tdsp-in-azure-ml.md).


![Fill in project information](./Images/instantiation-step3.jpg) 

The step-by-step data science workflow was as follows: 
        
**AT** REMOVED REPEATED SECTION ON DATA

### Modeling (Including Word2Vec word featurization/embedding) REMOVED NUMBERING
Modeling is the stage where we show how you can use the data downloaded in the previous section for training your own word embedding model and use it for other downstream tasks. Although we are using the Medline data, the pipeline to generate the embeddings is generic and can be reused to train word embeddings for any other domain. For embeddings to be an accurate representation of the data, it is essential that the word2vec is trained on a large amount of data.
Once we have the word embeddings ready, we can train a deep neural network that uses the learned embeddings to initialize the embedding layer. We mark the embedding layer as non-trainable but that is not mandatory. The training of the word embedding model is unsupervised and hence we are able to take advantage of unstructured texts. However, to train the supervised entity recognition model we need labeled data. The more the better.


#### [Featurizing/Embedding Words with Word2Vec](./Code/02_Modeling/01_FeatureEngineering/ReadMe.md)
Word2Vec is the name given to a class of neural network models that, given an unlabeled training corpus, produce a vector for each word in the corpus that encodes its semantic information. These models are simple neural networks with one hidden layer. The word vectors/embeddings are learned by backpropagation and stochastic gradient descent. There are two types of word2vec models, namely, the Skip-Gram and the continuous-bag-of-words models. Since we are using the MLlib's implementation of word2vec supports Skip-gram, we briefly describe that model here. For more details see [this paper](https://arxiv.org/pdf/1301.3781.pdf).

![Skip Gram Model](./Images/skip-gram.png)

The Skip-Gram model uses Hierarchical Softmax and Negative sampling to optimize the performance. Hierarchical SoftMax (H-SoftMax) is an approximation inspired by binary trees. H-SoftMax essentially replaces the flat SoftMax layer with a hierarchical layer that has the words as leaves. This allows us to decompose calculating the probability of one word into a sequence of probability calculations, which saves us from 
having to calculate the expensive normalization over all words. Since a balanced binary tree has a depth of log2(|V|)log2(|V|) (V is the Vocabulary), we only need to evaluate at most log2(|V|)log2(|V|) nodes to obtain the final probability of a word. The probability of a word w given its context c is then simply the product of the probabilities of taking right and left turns respectively that lead to its leaf node. We can build a Huffman Tree based on the frequency of the words in the dataset to ensure that more frequent words get shorter representations. Refer to this [blog](http://sebastianruder.com/word-embeddings-softmax/) for further information.
The image above was taken from [here](https://ahmedhanibrahim.wordpress.com/2017/04/25/thesis-tutorials-i-understanding-word2vec-for-word-embedding-i/)

**AT** PERHAPS THE ABOVE PARAGRAPH IS TOO MUCH DETAIL? TO ME, IT WOULD BE MORE INTERESTING TO READ HOW SKIM-GRAM TRAINS THE WORD EMBEDDINGS. PERHAPS JUST PROVIDE THE LINKS TO ARTICLES ON HOW H-SOFTMAX OPTIMIZES TRAINING.

Once we have the embeddings, we would like to visualize them and see the relationship between semantically similar words. 

![W2V similarity](./Images/w2v-sim.png)

There are two different ways of visualizing the embeddings. The first uses Principle Component Analysis (PCA) to project the high dimensional vector to a 2-D vector space. This leads to a significant loss of information and the visualization is not as accurate. The second is to use PCA with t-SNE. t-SNE is a nonlinear dimensionality reduction technique that is well-suited for embedding high-dimensional data into a space of two or three dimensions, which can then be visualized in a scatter plot. Refer to this [link](https://distill.pub/2016/misread-tsne/) for more details on t-SNE. Refer [./Code/02_Modeling/01_FeatureEngineering/ReadMe.md](./Code/02_Modeling/01_FeatureEngineering/ReadMe.md) for details about the implementation.

**AT** I SUGGEST SIMPLIFYING THIS PARAGRAPH AS ABOVE BECAUSE I THINK IT'S TOO MUCH DETAIL. IF THE READER IS INTERESTED IN EXACTLY HOW T-SNE WORKS THEY CAN FOLLOW THE LINK PROVIDED

As you see below, t-SNE visualization provides more separation and potential clustering patterns. 


* Visualization with PCA

![PCA](./Images/pca.png)

* Visualization with t-SNE

![t-SNE](./Images/tsne.png)

* Points closest to "Cancer" (they are all subtypes of Cancer)

![Points closest to Cancer](./Images/nearesttocancer.png)

#### [Training the Neural Entity Extractor](Code/02_Modeling/02_ModelCreation/ReadMe.md)
Traditional neural network models suffer from a problem in that they treat each input and output as independent of the other inputs and outputs. This may not be a good idea for tasks such as machine translation, entity extraction, or any other sequence-to-sequence labeling tasks. Recurrent neural network models (RNNs) overcome this problem as they can pass information computed in  previous nodes to the next node. This property means the network has memory since it is able to use the previously computed information. The below picture represents this.

![RNN](./Images/rnn-expanded.png)

Vanilla RNNs suffer from the [Vanishing Gradient Problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem) in which they are not able to utilize all the information they have seen before. The problem becomes evident only when a large amount of context is required to make a prediction. But models like LSTM do not suffer from such a problem, in fact they are designed to remember long-term dependencies. Unlike vanilla RNNs that have a single neural network, the LSTMs have the interactions between four neural networks for each cell. Refer to this [excellent post](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) for a detailed explanation of how LSTM work.

![LSTM Cell](./Images/lstm-cell.png) **AT** IMAGE NOT SHOWING

In this sample, we show how to put together our own LSTM-based recurrent neural network and extract entities like Drugs, Diseases etc. from medical data. The first step is to obtain a large amount of labeled data. Most of the medical data contains lot of sensitive information about the person and hence are not publicly available. We rely on a combination of two different datasets that are publicly available. The first dataset is from Semeval 2013 - Task 9.1 (Drug Recognition) and the other is from BioCreative V CDR task. We combine and auto label these two datasets so that we can detect both drugs and diseases from medical texts and evaluate our word embeddings. See [here](./Code/02_Modeling/02_ModelCreation/ReadMe.md) for implementation details.

**AT** I MADE THE ABOVE PARAGRAPH LESS CONVERSATIONAL.

**AT** PERHAPS YOU COULD EXPLAIN A LITTLE ABOUT HOW THE AUTO-LABELING WORKS?

The model architecture that we have used in all experiments is presented below. The maximum sequence length parameter (613 here) is changes for different datasets.

![LSTM model](./Images/d-a-d-model.png)

#### Model evaluation
We use the evaluation script from the shared task [Bio-Entity Recognition Task at Bio NLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) to evaluate the precision, recall, and F1 score of the model. Below is the comparison of the results we get with the embeddings trained on Medline Abstracts and with embeddings trained on Google News. We clearly see that the in-domain model out-performs the generic model. Hence having a specific word embedding model rather than using a generic one is much more helpful. 

![Model Comparison 1](./Images/mc1.png)

We perform the evaluation of the word embeddings on other datasets in a similar fashion and see that the in-domain model is always better.

![Model Comparison 2](./Images/mc2.png)

![Model Comparison 3](./Images/mc3.png)

![Model Comparison 4](./Images/mc4.png)

All the training and evaluations reported here are done using Keras and TensorFlow. Keras also supports CNTK backend but it does not yet have all the functionalities for the bidirectional model. We therefore used a unidirectional model with CNTK backend to benchmark the results of CNTK model with that of TensorFlow. We obtain the results below:

![Model Comparison 5](./Images/mc5.png)

We also compare the performance of Tensorflow vs CNTK and see that CNTK performs as well as Tensorflow, both in terms of time taken per epoch (60 secs for CNTK and 75 secs for Tensorflow) and the number of entities detected. We are using the unidirectional layers for evaluation.

![Model Comparison 6](./Images/mc6.png)


### [Deployment](./Code/03_Deployment/ReadMe.md) **AT** REMOVED NUMBERINGs
We show how to deploy a web service on a cluster in the [Azure Container Service (ACS)](https://azure.microsoft.com/en-us/services/container-service/). The operationalization environment provisions Docker and Kubernetes in the cluster to manage the web service deployment. You can find further information on the operationalization process [here](model-management-service-deploy.md ).


## Conclusion & Next Steps

In this sample, we demonstrated how to train a word embedding model using Word2Vec on Spark and then use the embeddings obtained for training a neural network for entity extraction. We have shown the pipeline for the Bio-Medical domain but the pipeline is generic. With enough data, you can easily adapt the workflow presented here to a different domain.

## Contact

Feel free to contact Mohamed Abdel-Hady (mohamed.abdel-hady@microsoft.com) Debraj GuhaThakurta (debraj.guhathakurta@microsoft.com) with any question or comments.
