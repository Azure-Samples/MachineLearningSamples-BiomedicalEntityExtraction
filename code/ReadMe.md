# Biomedical Entity Recognition using TDSP Template


## Introduction

The aim of this real-world scenario is to highlight how to use Azure Machine Learning Workbench to solve a complicated NLP task such as entity extraction from unstructured text. Here are the key points:

1. How to train a neural word embeddings model on a text corpus of about 18 million PubMed abstracts using [Spark Word2Vec implementation](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec).
2. How to build a deep Long Short-Term Memory (LSTM) recurrent neural network model for entity extraction on a GPU-enabled Azure Data Science Virtual Machine (GPU DSVM) on Azure.
2. Demonstrate that domain-specific word embeddings model can outperform generic word embeddings models in the entity recognition task. 
3. Demonstrate how to train and operationalize deep learning models using Azure Machine Learning Workbench.

The following capabilities within Azure Machine Learning Workbench:

   * Instantiation of [Team Data Science Process (TDSP) structure and templates](how-to-use-tdsp-in-azure-ml.md).
   * Automated management of your project dependencies including the download and the installation.
   * Execution of code in Jupyter notebooks as well as Python scripts.
   * Run history tracking for Python files.
   * Execution of jobs on remote Spark compute context using HDInsight Spark 2.1 clusters.
   * Execution of jobs in remote GPU VMs on Azure.
   * Easy operationalization of deep learning models as web services on Azure Container Services.

## Use Case Overview
Biomedical named entity recognition is a critical step for complex biomedical NLP tasks such as: 
* Extraction of diseases, symptoms from electronic medical or health records.
* Drug discovery.
* Understanding the interactions between different entity types such as drug-drug interaction, drug-disease relationship and gene-protein relationship.

Our use case scenario focuses on how a large amount of unstructured data corpus such as MedLine PubMed abstracts can be analyzed to train a word embedding model. Then the output embeddings are considered as automatically generated features to train a neural entity extractor.

Our results show that the biomedical entity extraction model training on the domain-specific word embedding features outperforms the model trained on the generic feature type. The domain-specific model can detect 7012 entities correctly (out of 9475) with F1-score of 0.73 compared to 5274 entities with F1-score of 0.61 for the generic model.

The following figure shows the architecture that was used to process data and train models.

![Architecture](../docs/images/architecture.png)

## Data Description

### 1. Word2Vec model training data
MedLine is a biomedical literature database. We first downloaded the 2017 release of MedLine from [here](https://www.nlm.nih.gov/pubs/factsheets/medline.html). The data is publically available in the form of XML files on their [FTP server](https://ftp.ncbi.nlm.nih.gov/pubmed/baseline). There are 892 XML files available on the server and each of the XML files has the information of 30,000 articles. More details about the data collection step are provided in the [Data Acquisition and Understanding](./code/01_data_acquisition_and_understanding/ReadMe.md) section. The fields present in each file are 
        
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

### 2. LSTM model training data

The neural entity extraction model has been trained and evaluated on the following publically available datasets:

 * [Bio-Entity Recognition Task at BioNLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html)
 * [BioCreative V CDR task corpus](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
 * [SemEval 2013 - Task 9.1 (Drug Recognition)](https://www.cs.york.ac.uk/semeval-2013/task9/)

The training and test data are TSV files where each sentence is represented in the IOB-format scheme. In this scheme, each token is tagged with one of three special chunk tags, I (inside), O (outside), or B (begin). A token is tagged as B if it marks the beginning of an entity mention such as B-Drug. Subsequent tokens within the entity mention are tagged I suhc as I-Drug. All other tokens are tagged O.
```
Naloxone	B-Chemical
reverses	O
the	O
antihypertensive	B-Drug
effect	O
of	O
clonidine	B-Chemical
.	O
```
## Prerequisites

* An Azure [subscription](https://azure.microsoft.com/en-us/free/)
* Azure Machine Learning Workbench with a created workspace. See [installation guide](quick-start-installation.md). 

### Azure services
* To run this scenario with Spark cluster, provision [Azure HDInsight Spark cluster](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) (Spark 2.1 on Linux (HDI 3.6)) for scale-out computation. To process the full amount of MedLine abstracts discussed below, we recommend having a cluster with:
    * a head node of type [D13_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/) 
    * at least four worker nodes of type [D12_V2](https://azure.microsoft.com/en-us/pricing/details/hdinsight/). 

    * To maximize performance of the cluster, we recommend to change the parameters spark.executor.instances, spark.executor.cores, and spark.executor.memory by following the instructions [here](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-spark-sql) and editing the definitions in "custom spark defaults" section. 

* You can run the entity extraction model training locally on a [Data Science Virtual Machine (DSVM)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro) or in a remote Docker container in a remote DSVM.

* To provision DSVM for Linux (Ubuntu), follow the instructions [here](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-provision-vm). We recommend using [NC6 Standard (56 GB, K80 NVIDIA Tesla)](https://docs.microsoft.com/en-us/azure/machine-learning/machine-learning-data-science-linux-dsvm-intro).

### Python packages

All the required dependencies are defined into three .yml files under the scenario project folder:
* the [aml_config/myspark_conda_dependencies.yml](aml_config/myspark_conda_dependencies.yml) file: the dependencies defined in this file will be automatically provisioned for the word embedding model training runs against HDI cluster targets.
* the [aml_config/myvm_conda_dependencies.yml](aml_config/myvm_conda_dependencies.yml) file: the dependencies defined in this file will be automatically provisioned for the Keras deep learning model training runs against remote docker into remote VM targets. For runs into local VM as a target, you have to installs the defined dependencies manually from the CLI window.
* the [aml_config/scoring_conda_dependencies.yml](aml_config/scoring_conda_dependencies.yml) file: the dependencies defined in this file will be automatically provisioned for the scoring web service runs into ACS cluster. The main difference between this yml file and myvm_conda_dependencies.yml file is to install TensorFlow CPU version instead of TensorFlow GPU version that is used for training and testing.

 For details about the Conda environment file format, refer to [here](https://conda.io/docs/using/envs.html#create-environment-file-by-hand).
Here are the basic packages required to run this project:
* [TensorFlow with GPU support](https://www.tensorflow.org/install/)
* [TensorFlow CPU version](https://www.tensorflow.org/install/)
* [CNTK 2.0](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras) could be used as backend for Keras instead of TensorFlow
* [Keras](https://keras.io/#installation)
* NLTK
* Fastparquet

### Basic instructions for Azure Machine Learning (AML) Workbench
* [Overview](overview-what-is-azure-ml.md)
* [Installation](quick-start-installation.md)
* [Using TDSP](how-to-use-tdsp-in-azure-ml.md)
* [How to read and write files](how-to-read-write-files.md)
* [How to use Jupyter Notebooks](how-to-use-jupyter-notebooks.md)
* [How to use GPU](how-to-use-gpu.md)

## Scenario Structure
For the scenario, we use the TDSP project structure and documentation templates (Figure 1), which follows the [TDSP lifecycle](https://github.com/Azure/Microsoft-TDSP/blob/master/Docs/lifecycle-detail.md). The project is created based on the instructions provided [here](https://github.com/amlsamples/tdsp/blob/master/docs/how-to-use-tdsp-in-azure-ml.md).


![Fill in project information](../docs/images/instantiation-3.png) 
Figure 1. TDSP Template in AML Workbench.

### Configuration of execution environments

This project includes steps that run on two compute/execution environments: in Spark cluster and GPU-supported DSVM. We start with the description of the dependencies required for both environments. 

In the next steps, we connect execution environments to an Azure account. Open a Command Line Interface (CLI) window by clicking File menu in the top left corner of AML Workbench and choosing "Open Command Prompt". Then run in CLI the following command:
```
    az login
```
You get a message
```
    To sign in, use a web browser to open the page https://aka.ms/devicelogin and enter the code <code> to authenticate.
```
Go to this web page, enter the code and sign into your Azure account. After this step, run in CLI
```
    az account list -o table
```
and find the subscription ID of the Azure subscription that has your AML Workbench Workspace account. Finally, run in CLI
```
    az account set -s <subscription ID>
```
to complete the connection to your Azure subscription.

In the next two sections we show how to complete the configuration of the remote Docker container and the Spark cluster environments.

#### Configuration of remote Docker container

To install the required packages in the Docker image, we created the following myvm_conda_dependencies.yml file stored in the aml_config directory of this project:

    name: project_environment    
    dependencies:
    - python=3.5.2
    # ipykernel is required to use the remote/docker kernels in Jupyter Notebook.
    - ipykernel=4.6.1
    - tensorflow-gpu
    - nltk
    - requests
    - lxml
    - unidecode
    - pip:
        # This is the operationalization API for Azure Machine Learning. Details:
        # https://github.com/Azure/Machine-Learning-Operationalization
        - azure-ml-api-sdk==0.1.0a6
        - h5py==2.7.0
        - matplotlib
        - fastparquet
        - keras
        - azure-storage

 To set up a remote Docker container, run the following command in the CLI:
```
    az ml computetarget attach remotedocker --name myvm --address <IP address> --username <username> --password <password>
```
with IP address, user name and password for the DSVM. The IP address of a DSVM can be found in the Overview section of your DSVM page in Azure portal:

![VM IP](../docs/images/vm_ip.png)

This command creates two files under the aml_config folder: myvm.compute and myvm.runconfig. Then modify the myvm.runconfig file as follows:
 
1. Set the PrepareEnvironment flag to true.
2. Modify the CondaDependenciesFile parameter to point to the myspark_conda_dependencies.yml file.

```
ArgumentVector:
- $file
CondaDependenciesFile: aml_config/myvm_conda_dependencies.yml
EnvironmentVariables: null
Framework: PySpark
PrepareEnvironment: false
SparkDependenciesFile: aml_config/spark_dependencies.yml
Target: myvm
TrackedRun: true
UseSampling: true
```
Then, you will be asked to run the following command.

```
az ml experiment prepare -c myvm
```

#### Configuration of Spark cluster

To install the required packages into the Spark cluster nodes, we created the following myspark_conda_dependencies.yml file stored in the aml_config directory of this project:

    name: project_environment    
    dependencies:
    - python=3.5.2
    # ipykernel is required to use the remote/docker kernels in Jupyter Notebook.
    - ipykernel=4.6.1    
    - nltk
    - requests
    - lxml
    - unidecode
    - pip:
        # This is the operationalization API for Azure Machine Learning. Details:
        # https://github.com/Azure/Machine-Learning-Operationalization
        - azure-ml-api-sdk==0.1.0a6
        - h5py==2.7.0
        - matplotlib
        - fastparquet        
        - azure-storage

To set up Spark environment, run the following command in the CLI:
```
    az ml computetarget attach cluster --name myspark --address <cluster name>-ssh.azurehdinsight.net  --username <username> --password <password>
```
with the name of the cluster, cluster's SSH user name and password. The default value of SSH user name is `sshuser`, unless you changed it during provisioning of the cluster. The name of the cluster can be found in the Properties section of your cluster page in Azure portal:

![Cluster name](../docs/images/cluster_name.png)

This command creates two files myspark.compute and myspark.runconfig under aml_config folder. Then modify the myspark.runconfig file as follows:

1. Set the PrepareEnvironment flag to true.
2. Modify the CondaDependenciesFile parameter to point to the myspark_conda_dependencies.yml file.

```
ArgumentVector:
- $file
CondaDependenciesFile: aml_config/myspark_conda_dependencies.yml
EnvironmentVariables: null
Framework: PySpark
PrepareEnvironment: true
SparkDependenciesFile: aml_config/spark_dependencies.yml
Target: dl4nlp-cluster
TrackedRun: true
UseSampling: true
```
Then, you will be asked to run the following command.
```
az ml experiment prepare -c myspark
```

The step-by-step data science workflow is as follows:
### 1. [Data Acquisition and Understanding](./01_data_acquisition_and_understanding/ReadMe.md)
### 2. [Modeling](./02_modeling/ReadMe.md)
#### 2.1. [Feature engineering](./02_modeling/01_feature_engineering/ReadMe.md)
#### 2.2. [Train the neural entity extractor](./02_modeling/02_model_creation/ReadMe.md)
#### 2.3. [Model evaluation](./02_modeling/03_model_evaluation/ReadMe.md)
### 3. [Deployment](./03_deployment/ReadMe.md)


## Conclusion

This use case scenario demonstrates how to train a domain-specific word embedding model using Word2Vec algorithm on Spark and then use the extracted embeddings as features to train a deep neural network for entity extraction. We have applied the training pipeline on the biomedical domain. However, the pipeline is generic enough to be applied to detect custom entity types of any other domain. You just need enough data and you can easily adapt the workflow presented here for a different domain.

## References

* Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. 2013a. Efficient estimation of word representations in vector space. In Proceedings of ICLR.
* Tomas Mikolov, Ilya Sutskever, Kai Chen, Greg S Corrado, and Jeff Dean. 2013b. Distributed representations of words and phrases and their compositionality. In Proceedings of NIPS, pages 3111â3119.
* Billy Chiu, Gamal Crichton, Anna Korhonen and Sampo Pyysalo. 2016. [How to Train Good Word Embeddings for Biomedical NLP](http://aclweb.org/anthology/W/W16/W16-2922.pdf), In Proceedings of the 15th Workshop on Biomedical Natural Language Processing, pages 166â174.
* Isabel Segura-Bedmar, V´ictor Su´arez-Paniagua, Paloma Mart´inez. 2015. [Exploring Word Embedding for Drug Name Recognition](https://aclweb.org/anthology/W/W15/W15-2608.pdf), In Proceedings of the Sixth International Workshop on Health Text Mining and Information Analysis (Louhi), pages 64â72, Lisbon, Portugal, 17 September 2015. 
* [Vector Representations of Words](https://www.tensorflow.org/tutorials/word2vec)
* [Recurrent Neural Networks](https://www.tensorflow.org/tutorials/recurrent)
* [Problems encountered with Spark ml Word2Vec](https://intothedepthsofdataengineering.wordpress.com/2017/06/26/problems-encountered-with-spark-ml-word2vec/)
* [Spark Word2Vec: lessons learned](https://intothedepthsofdataengineering.wordpress.com/2017/06/26/spark-word2vec-lessons-learned/)



## Contributing

This project welcomes contributions and suggestions. Most contributions require you to agree to a Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us the rights to use your contribution. For details, visit https://cla.microsoft.com.
When you submit a pull request, a CLA-bot will automatically determine whether you need to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the instructions provided by the bot. You will only need to do this once across all repos using our CLA.
This project has adopted the Microsoft Open Source Code of Conduct. For more information see the Code of Conduct FAQ or contact opencode@microsoft.com with any additional questions or comments.
