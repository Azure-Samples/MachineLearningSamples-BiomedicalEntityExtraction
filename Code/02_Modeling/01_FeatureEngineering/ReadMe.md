## [Train Word2Vec Word Embedding Model](2_Train_Word2Vec.ipynb)
This [Notebook](2_Train_Word2Vec.ipynb) covers how you can train, evaluate and visualize word embeddings using **[Word2Vec](https://arxiv.org/pdf/1301.3781.pdf)** implementaion from **[MLLib](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#word2vec)** 
in **Spark**. The MLLib function for word2vec is based on a continuos skip-gram model that tries to predict the context words given a word. To optimse the performance this implementation uses hierarchical softmax. H-SoftMax 
essentially replaces the flat SoftMax layer with a hierarchical layer that has the words as leaves. This allows us to decompose calculating the probability of one word into a sequence of probability calculations, 
which saves us from having to calculate the expensive normalization over all words. The algorithm has several hyper-parameters which can be tuned to obtain better performance. These are windowSize, vectorSize etc. (We 
define the meaning of each parameter in step 5). results for hyper parameter tuning are present in the end. Let's begin extracting word embeddings for Bio-medical terms.

We are going to use the XML files we downloaded in [section 1](../../01_DataPreparation/1_Download_and_Parse_Medline_Abstracts.ipynb)

- Step 1: Import the required libraries and point the path to the directory where you downloaded the XML files.

- Step 2: Combine all the data from these XMLs into a single dataframe and store this new dataframe in the Parquet format so that we can read it quickly in the next runs. This may take about 15-30 mins depending on the size of your 
spark cluster.

- Step 3: (Optional) If you have a dictionary of Words that you care about and want to exclude any abstracts that do not contain words from this dictionary you can uncomment the filtering code.

- Step 4: Do some basic preprocessing on the abstracts and load test sets for evaluation. 

- Step 5: Train Word2Vec model using the Continuous Skip-gram Model (We use the MLlib Word2Vec function for this). Set the parameters of the model based on your requirements 

    * windowSize (number of words of the left and right eg. window size of 2 means 2 words to the left and 2 to the right of the current word.) 
    * vectorSize (size of the embeddings),
    * minCount (minium number of occurences of a word to be included in the output)
    * numPartitions (number of partitions used for training, keep a small number for accuracy)

- Step 6: Save the Word Embeddings in Parquet format

#### Notes:

- Hyper-Parameter Tuning: The training time of the Word2Vec algorithm depends on the hyper-parameter values as well as the size of the Spark cluster.
    *  We also see having a larger mincount is giving better results but it is also decreasing the coverage over the test set, hence its advisable to keep micount as a low number.
    * The numbers here are reported on a spark cluster having 11 worker nodes with each worker node with 4 cores. The runtime for most of the evaluations is under 30 mins because of the large number of partitions. 
    * If speed is the main concern then number of partitions should be as high as possible (but less than the total number of cores), however if the concer is accuracy the number of partitions should be a lower number (it will take more time).

- Memory Issues

 While working with spark there might be a few places where you may get Memory Exceptions. For example, while downloading the XML files it is advisable to continuosly store the data rather than wait for storing after completing the entire processing. Another place is using Word2vec with very small mincount, high vector size, low number of partitions. If the dataset you are working with is as huge as the Medline then it is advisable to test the performance on a sampled dataset like (10% or so) and then scale the parameters.
