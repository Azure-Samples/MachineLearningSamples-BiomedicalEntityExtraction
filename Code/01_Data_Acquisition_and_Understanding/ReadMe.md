## **Data Preparation**
This section describes how to download the [MEDLINE](https://www.nlm.nih.gov/pubs/factsheets/medline.html) abstracts from the website using **Spark**. We are using HDInsight Spark 2.1 on Linux (HDI 3.6).
The FTP server for Medline has about 812 XML files where each file contains about 30,000 abstracts. Below you can see the fields present in the XML files. We use the Abstracts extracted from the XML files to train the word embedding model.


### [Downloading and Parsing Medline Abstracts](1_Download_and_Parse_Medline_Abstracts.ipynb)
The [Notebook]((1_Download_and_Parse_Medline_Abstracts.ipynb) **AT** LINK DOESN'T WORK describes how to download the local drive of the head node of the Spark cluster. Since the data is big (about 30 Gb), it might take a while to download. We parse the XML files as we download them. We use a publicly available [Pubmed Parser](https://github.com/titipata/pubmed_parser) to parse the downloaded XMLs and save them in a tab separated file (TSV). The parsed XMLs are stored in a local folder on the head node (you can change this by specifying a different location in the notebook). The parse XML returns the following fields:

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

**Notes**:
- There are more that 800 XML files that are present on the Medline ftp server. The code in the notebook downloads them all. But you can change that in the last cell of the notebook (e.g. download only a subset by reducing the counter).
- With using Tab separated files: The Pubmed parser adds a new line for every affiliation. This may cause the TSV files to become unstructured. To **avoid** this we explicitly remove the new line from the affiliation field.
- To install unidecode, you can use [script action](https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-hadoop-customize-cluster-linux) on your Spark Cluster. Add the following lines to your script file (.sh). 
You can install other dependencies in a similar way

        #!/usr/bin/env bash
        /usr/bin/anaconda/bin/conda install unidecode

- The egg file needed to run the Pubmed Parser is also included in the repository. **AT** I DO NOT SEE THIS .EGG FILE IN THE REPO

