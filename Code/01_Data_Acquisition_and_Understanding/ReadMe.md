### 1. [Data Acquisition and Understanding](https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/Code/01_Data_Acquisition_and_Understanding/ReadMe.md)

The raw MEDLINE corpus has a total of 27 million abstracts where about 10 million articles have an empty abstract field. Azure HDInsight Spark is used to process big data that cannot be loaded into the memory of a single machine as a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). First, the data is downloaded into the Spark cluster. Then the following steps are executed on the [Spark DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html): 
* parse the XML files using Medline XML Parser
* preprocess the abstract text including sentence splitting, tokenization and case normalization.
* exclude articles where abstract field is empty or has short text 
* create the word vocabulary from the training abstracts
* train the word embedding neural model. For more details, refer to [GitHub code link](https://github.com/Azure/MachineLearningSamples-BiomedicalEntityExtraction/blob/master/Code/01_DataPreparation/ReadMe.md) to get started.


After parsing XML files, data has the following format: 

![Data Sample](./Images/datasample.png)

Other datasets, which are being used for training and evaluation of the Neural Entity Extractor have been include in the corresponding folder. To obtain more information about them, you could refer to the following corpora:
 * [Bio-Entity Recognition Task at BioNLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html)
 * [BioCreative V CDR task corpus](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
 * [Semeval 2013 - Task 9.1 (Drug Recognition)](https://www.cs.york.ac.uk/semeval-2013/task9/)
 

**Notes**:
- There are more that 800 XML files that are present on the Medline ftp server. The code in the Notebook downloads them all. But you can change that in the last cell of the notebook (e.g. download only a subset by reducing the counter).
- With using Tab separated files: The Pubmed parser add a new line for every affiliation. This may cause the TSV files to become unstructured. To **avoid** this we explicitly remove the new line from the affiliation field.
- To install unidecode, you can use [script action](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-customize-cluster-linux) on your Spark Cluster. Add the following lines to your script file (.sh). 
You can install other dependencies in a similar way
        #!/usr/bin/env bash
        /usr/bin/anaconda/bin/conda install unidecode
- The egg file needed to run the Pubmed Parser is also included in the repository.

