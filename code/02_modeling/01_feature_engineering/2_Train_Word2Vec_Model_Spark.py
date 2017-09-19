
# coding: utf-8

# ## Train, Evaluate and Visualize the Word Embeddings
# In this Notebook we detail the process of how to train a word2vec model on
# the Medline Abstracts and obtaining the word embeddings for the biomedical
# terms.  This is the first step towards building an Entity Extractor.  We are
# using the Spark's <a href = "https://spark.apache.org/mllib/">MLLib</a>
# package to train the word embedding model.  We also show how you can test the
# quality of the embeddings by an intrinsic evaluation task along with
# visualization.
# <br>
# The Word Embeddings obtained from spark are stored in parquet files with gzip
# compression.  In the next script, we show how to aggregate the distributed
# word embeddings into a single pickle file then to load them into memory for the
# feature extraction step.
# Please increase the memory size before starting the training using the Ambari dashboard: spark.driver.memory 4000m
from pyspark import SparkConf
from pyspark import SparkContext
from pyspark.sql import SQLContext
from pyspark.sql import SparkSession
from pyspark.sql.functions import UserDefinedFunction
from pyspark.ml.feature import Word2Vec, Word2VecModel
from pyspark.ml.feature import Tokenizer, RegexTokenizer, StopWordsRemover
from pyspark.sql.types import *
import matplotlib.pyplot as plt
import numpy as np
import datetime

import os
import sys

spark = SparkSession \
    .builder \
    .appName("Train word embeddings model") \
    .getOrCreate()

# <b> Setup the paths where the TSV files are located <b>

parse_results_remote_dir = os.path.join('wasb:///', 'pubmed_data', 'tsv_files')


# ### Step 2

# ### Read in Pubmed Abstracts from csv
# <br> This cell iterates over all the parsed pubmed files and combines them
# into a single dataframe.  <br>It then writes this new dataframe to a parquet
# file for faster access in Subsequent runs.<br><br>
# Note: <ol><li>Ideally you should run the below cell only <b>once</b> and use
# the parquet file for subsequent runs.</li>
# <li> This cell saves a parquet files after reading every 100 files.  If this
# block fails for some reason, you can resume from the latest file.

# In[5]:
timestart = datetime.datetime.now()
num_xml_files = 892 
batch_size =50
pubmed_tsv_file = os.path.join(parse_results_remote_dir, 'batch#{}.tsv'.format(1))   
print("Reading file {}".format(pubmed_tsv_file))     
abstracts_batch_df = spark.read.csv(path=pubmed_tsv_file, header=True, inferSchema=True, sep = "\t")

print("\tAdding {} records ...".format(abstracts_batch_df.count()))
abstracts_full_df = abstracts_batch_df

for i in range(1 + batch_size, num_xml_files + 1, batch_size):  
    try:
        pubmed_tsv_file = os.path.join(parse_results_remote_dir, 'batch#{}.tsv'.format(i))   
        print("Reading file {}".format(pubmed_tsv_file))     
        abstracts_batch_df = spark.read.csv(path=pubmed_tsv_file, header=True, inferSchema=True, sep = "\t")

        print("\tAdding {} records ...".format(abstracts_batch_df.count()))
        
        abstracts_full_df = abstracts_full_df.union(abstracts_batch_df)
       
    except:
        print("Skipped" + str(i))

'''
# uncomment the following for quick testing
# train the model on a small sample
sample_rate = 0.001
print("Take {} random sample".format(sample_rate))
abstracts_full_df = abstracts_full_df.sample(True, sample_rate)
'''

abstracts_full_df.printSchema()
print("abstracts_full_df.count() = {}".format(abstracts_full_df.count()))
print("abstracts_full_df.head() = {}".format(abstracts_full_df.head()))

# pubmedAbstracts_full_df.select("title","abstract").repartition(300).write.mode("overwrite").parquet(pubmedTitAbs_path + "full")

# PRINT HOW MUCH TIME IT TOOK TO RUN THE CELL
timeend = datetime.datetime.now()
timedelta = round((timeend - timestart).total_seconds() / 60, 2)
print("Time taken to execute above cell: " + str(timedelta) + " mins")

abstracts_full_df2 = abstracts_full_df


# ### Step 2

# <b> Preprocess the Abstracts </b><br>
# We do some basic pre-processing like converting words to lowercase and
# removing separators other than - and _.  <br>
# We then split the words into tokens.
# https://spark.apache.org/docs/latest/ml-features.html#tokenizer

# In[9]:
from pyspark.sql.functions import regexp_replace, trim, col, lower, udf
from pyspark.ml.feature import Tokenizer, RegexTokenizer
from pyspark.ml.feature import StopWordsRemover

timestart = datetime.datetime.now()

print("abstracts_full_df2.head() = {}".format(abstracts_full_df2.head()))

# Convert the content to Lower Case
print("Converting the abstarct to Lower Case ... ")
abstracts_full_df3 = abstracts_full_df2.withColumn("abstractNew", lower(col("abstract"))).\
    withColumn("abstractNew", regexp_replace("abstractNew", '[^\w-_ ]', ""))

abstracts_full_df3.printSchema()
# print("abstracts_full_df3.head() = {}".format(abstracts_full_df3.head()))

# Tokenize the Abstracts
print("tokenizating the abstracts... ")
tokenizer = Tokenizer(inputCol="abstractNew", outputCol="words")
remover = StopWordsRemover(inputCol="words", outputCol="filtWords")

abstracts_full_df4 = tokenizer.transform(abstracts_full_df3)

print("After tokenization: ")
abstracts_full_df4.printSchema()
print("abstracts_full_df4.count() = {}".format(abstracts_full_df4.count()))
# print("abstracts_full_df4.head() = {}".format(abstracts_full_df4.head()))

# PRINT HOW MUCH TIME IT TOOK TO RUN THE CELL
timeend = datetime.datetime.now()
timedelta = round((timeend - timestart).total_seconds() / 60, 2)
print("Time taken to execute above cell: " + str(timedelta) + " mins")


# ### Step 5

# #### Train <a href =
# "https://spark.apache.org/docs/2.1.1/api/java/org/apache/spark/mllib/feature/Word2Vec.html">
# Word2Vec Model</a> (it takes some time to run depending on the data size)
# <ul>
# <li> It runs in slighlty over 15 minutes when run on 15 million abstracts
# with a window size of 5, vector size of 50 and mincount of 400.</li><li> This
# performance is on a spark cluster with 11 worker nodes each with 4
# cores.</li>
# <li> The parameter values of 5 for window size, 50 for vector size and
# mincount of 400 work well for the Entity Recognition Task.</li><li> However
# optimal performance (Rho = 0.5632) during the Intrinsic Evaluation is
# achieved with a window size of 30, vector size of 100 and mincount of 400.
# </li><li>This difference can be attributed to the fact that a bigger window
# size does not help the Entity Recognition task and a simpler model is
# preferred.</li>
# <li> To speed up the evaluation of word2vec change number of partitions to a
# higher value (&lt; number of cores available), but this may decrease the
# accuracy of the model.
# </ul>
# <br> For the cell below we are using a cluster size of 4 worker nodes each
# with 4 cores.

timestart = datetime.datetime.now()
model = None
window_size = 5
vector_size = 50
min_count =400
print("Start training the model ...")
word2Vec = Word2Vec(windowSize = window_size, vectorSize = vector_size, minCount=min_count, numPartitions=10, inputCol="words", outputCol="result")
model = word2Vec.fit(abstracts_full_df4)

# PRINT HOW MUCH TIME IT TOOK TO RUN THE CELL
timeend = datetime.datetime.now()
timedelta = round((timeend - timestart).total_seconds() / 60, 2)
print("model.getVectors().count() = {}".format(model.getVectors().count()))
print("Time taken to execute above cell: " + str(timedelta) + " mins")


# Save the model for future use

#Change the path to the location where you want to save your model
# model_file = "wasb:///Models/w2vmodel_entiredataset_vs_{}_ws_{}_mc_{}.p".format(vector_size, window_size, min_count)
# print("Saving the model into binary format {}".format(model_file))
# sc = SparkContext.getOrCreate()
# model.save(path=model_file)


# print("Loading the model ")
# new_model = Word2VecModel.load(sc, model_file)
# embedding_dic = new_model.getVectors()
# print(embedding_dic.keys)

'''
# <b> Manually Evaluate Similar words by getting nearest neighbours </b>

# In[13]:
model.findSynonyms("cancer", 20).select("word").head(20) #Returns types of Cancers, hormones responsible for Cancer etc.


# In[14]:
model.findSynonyms("brain", 20).select("word").head(20)# Returns Different Parts of the Brain
'''

# <b>Store the word vectors in a SQL table</b>
df = model.getVectors()
df.printSchema()
print("vocabulary size = {}".format(df.count()))

# REGISTER Vectors DF IN SQL-CONTEXT
df.createOrReplaceTempView("word2vec")
spark.sql("show tables").show()

'''
# <br><br><b> Load the test data for Intrinsic Evaluation </b>
# <br>Test Set Details
# <ul>
# <li>Contains 588 concept (drugs, diseases, symptoms) pairs manually rated for
# semantic relatedness.</li>
# <li>The Ratings provided are on an arbitrary scale.</li>
# </ul>
# <br>A few details about the Spearman correlation
# <ul>
# <li>Spearman Correlation (ρ) is a nonparametric measure of rank correlation.
# </li>
# <li>It assesses how well the relationship between two variables(pairs ranked
# by humans VS pairs ranked by our model) can be described using a monotonic
# function.  </li>
# <li>It compares ranking between variables regardless of their scale</li>
# </ul>
# <br>

# #### Load the Test Set

# In[10]:

#Change the path to the location of UMNSRS-rel.txt
UMNSRS_rel_file = "wasb:///data/UMNSRS-rel.txt"
UMNSRS_rel = spark.read.csv(path = UMNSRS_rel_file, header=True, inferSchema=True, sep = "\t")
UMNSRS_rel.printSchema()

# REGISTER UMNSRS_rel DF IN SQL-CONTEXT
UMNSRS_rel.createOrReplaceTempView("dict")
spark.sql("show tables").show()


# <b> Load the words in from the test set </b>
# <br>This loads a dictionary of all the words present in the test set.  We'll
# use this later while evaluating the word embeddings.
#

# In[11]:

#Change the path to the location of Test_words.txt
test_words_file = "wasb:///data/Test_words.txt"
test_words = spark.read.csv(path = test_words_file, header=True, inferSchema=True, sep = "\n")
test_words.printSchema()

# REGISTER DF IN SQL-CONTEXT
test_words.createOrReplaceTempView("test_dict")
spark.sql("show tables").show()


# ### Step 6
# The below cell uses SQL magic(%%).  Read more <a href =
# "https://docs.microsoft.com/en-us/azure/hdinsight/hdinsight-apache-spark-jupyter-notebook-kernels#parameters-supported-with-the-sql-magic">here</a>

# In[18]:
get_ipython().run_cell_magic(u'sql', u'-q -o rel_file', u'select * from dict')


# <b> Obtain the Word Vector for all the words in the test set for Intrinsic
# Evaluation of the Embeddings </b>

# In[16]:
get_ipython().run_cell_magic(u'sql', u'-q -o sqlResultsPD', u'select wv.word, wv.vector from word2vec wv\ninner join test_dict d on wv.word = d.words_test')


# <b> Intrinsic Evaluation </b><br>
# We are using <a href =
# "https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient">
# Spearman Correlation </a> to evaluate the performance of similarity from our
# embeddings against the similarity given by humans.

# In[19]:
get_ipython().run_cell_magic(u'local', u'', u'import numpy as np\nfrom scipy.spatial.distance import cosine as cos\nfrom scipy.stats import spearmanr\nimport datetime\ntimestart = datetime.datetime.now()\n\nall_words = np.array(sqlResultsPD.word)\nall_vecs = np.array(sqlResultsPD.vector)\n\nrel_file = np.array(rel_file)\nrel_file_len = len(rel_file)\n\nsimilarity_actual = []\nsimilarity_predicted = []\n\nfound = 0\nfor i in range(rel_file_len):\n    word1 = rel_file[i][0]\n    word2 = rel_file[i][1]\n    vec1_pos = -1\n    for j in range(len(all_words)):\n        if all_words[j] == word1:\n            vec1_pos = j\n            break\n    vec2_pos = -1\n    for j in range(len(all_words)):\n        if all_words[j] == word2:\n            vec2_pos = j\n            break\n\n    if vec1_pos >= 0 and vec2_pos >= 0:\n        similarity_actual.append(rel_file[i][2])\n        found += 1        \n        vec1 = all_vecs[vec1_pos][\'values\']\n        vec2 = all_vecs[vec2_pos][\'values\']\n        val = 1 - cos(vec1, vec2)\n        similarity_predicted.append(val)\n\nsimilarity_actual = np.array(similarity_actual)\nsimilarity_predicted = np.array(similarity_predicted)\nrho, count = spearmanr(similarity_actual, similarity_predicted)\nprint("rho = ", rho)\nprint(found, rel_file_len, found*100/rel_file_len)\n\n# PRINT HOW MUCH TIME IT TOOK TO RUN THE CELL\ntimeend = datetime.datetime.now()\ntimedelta = round((timeend-timestart).total_seconds() / 60, 2)\nprint ("Time taken to execute above cell: " + str(timedelta) + " mins");')


# In[124]:
get_ipython().run_cell_magic(u'sql', u'', u'drop table dict')
'''

# ### Step 7

# <b> Function to split the vectors from the dataframe into columns</b><br>This
# makes it easier to save the Embeddings

# In[20]:
from pyspark.sql.types import DoubleType
from pyspark.sql.functions import lit, udf

def get_ith_column_(v, i):
    try:
        return float(v[i])
    except ValueError:
        return None

get_ith_column = udf(get_ith_column_, DoubleType())


# <b> Split the Word Embeddings into Columns </b>

# In[21]:
from pyspark.ml.linalg import Vectors
df_1 = df
for i in range(vector_size):
    # add new column to the Spark dataframe
    df_1 = df_1.withColumn("col" + str(i), get_ith_column("vector", lit(i)))

df_1 = df_1.drop("vector")
df_1.printSchema()

# <b> Save the Word Embeddings in the parquet format</b><br> Remember to use
# the compression as gzip since the method we will use to load the embeddings
# by parsing the Parquet files currently supports only gzip compression.<br>
# Note: If you want to load the word embeddings in python for visualization or
# for performing some down stream tasks, checkout the first section of the next
# Notebook for a simple method of reading the Word Embeddings from parquet
# files and storing it as a dictionary
#
# You may also be interested in the following posts
# (1) Problems encountered with Spark ml Word2Vec
# https://intothedepthsofdataengineering.wordpress.com/2017/06/26/problems-encountered-with-spark-ml-word2vec/ 
# (2) Spark Word2Vec: lessons learned 
# https://intothedepthsofdataengineering.wordpress.com/2017/06/26/spark-word2vec-lessons-learned/
# In[22]:

#Change the path to the location where you want to store the Embeddings
model_file = "wasb:///Models/word2vec_pubmed_model_vs_{}_ws_{}_mc_{}_parquet_files".\
    format(vector_size, window_size, min_count)
# print("Saving the model into binary format {}".format(model_file))
df_1.repartition(1000).write.mode("overwrite").parquet(model_file, compression='gzip')

# model_file = "wasb:///Models/word2vec_pubmed_model_vs_{}_ws_{}_mc_{}_tsv_files".\
#     format(vector_size, window_size, min_count)

# print("Writing {} records to file {} .....".format(df_1.count(), model_file))        
# df_1.repartition(1000).write.\
#             format("com.databricks.spark.csv").\
#             option("header", "true").\
#             option("delimiter", "\t").\
#             save(model_file,  mode='overwrite')
