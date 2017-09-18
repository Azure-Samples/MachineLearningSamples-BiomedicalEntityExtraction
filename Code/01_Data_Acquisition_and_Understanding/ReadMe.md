### 1. [Data Acquisition and Understanding](1_Download_and_Parse_XML_Spark.py)

The [companion script](1_Download_and_Parse_XML_Spark.py) covers how to download and parse the MEDLINE corpus.

The raw MEDLINE corpus has a total of 27 million abstracts where about 10 million articles have an empty abstract field. Azure HDInsight Spark is used to process big data that cannot be loaded into the memory of a single machine as a [Pandas DataFrame](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html). First, the data is downloaded into the Spark cluster. Then the following steps are executed on the [Spark DataFrame](https://spark.apache.org/docs/latest/sql-programming-guide.html): 
* parse the XML files using Medline XML Parser
* preprocess the abstract text including sentence splitting, tokenization and case normalization.
* exclude articles where abstract field is empty or has short text 
* create the word vocabulary from the training abstracts
* train the word embedding neural model. For more details, refer to [GitHub code link](../02_modeling/01_feature_engineering/ReadMe.md) to get started.

After parsing the Medline XML files, each data record has the following format: 

![Data Sample](../../docs/images/datasample.png)

The neural entity extraction model has been trained and evaluated on publiclly available datasets. To obtain a detailed description about these datasets, you could refer to the following sources:
 * [Bio-Entity Recognition Task at BioNLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html)
 * [BioCreative V CDR task corpus](http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/)
 * [Semeval 2013 - Task 9.1 (Drug Recognition)](https://www.cs.york.ac.uk/semeval-2013/task9/)
 

**Notes**:
- There are more that 800 XML files that are present on the Medline ftp server. The shared code downloads them all which takes a long time. If you just ned to test the code, you can change that and download only a subsample.
- The source code of the PubMed Parser is also included in the repository.

### Next Step
2. [Modeling](./code/02_modeling/ReadMe.md)