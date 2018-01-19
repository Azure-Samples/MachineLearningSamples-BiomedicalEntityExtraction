
# coding: utf-8

# ## Download and Parse MEDLINE Abstracts
# This Notebook describes the way you can download and parse the publically
# available Medline Abstracts.  There are about 812 XML files that are
# available on the ftp server.  Each XML file conatins about 30,000 Document
# Abstracts.
# <ul>
# <li> First we download the Medline XMLs from their FTP Server and store them
# in a local directory on the head node of the Spark Cluster </li>
# <li> Next we parse the XMLs using a publically available Medline Parser and
# store the parsed content in Tab separated files on the container associated
# with the spark cluster.  </li>
# </ul>
# <br>Note: This Notebook is meant to be run on a Spark Cluster.  If you are
# running it through a jupyter notebbok, make sure to use the PySpark Kernel.

# #### Using the Parser
# Download and install the pubmed_parser library into the spark cluster nodes.
# You can us the egg file available in the repo or produce the .egg file by
# running<br>
# <b>python setup.py bdist_egg </b><br>
# in repository and add import for it.  The egg file file can be read from the
# blob storage.  Once you have the egg file ready you can put it in the
# container associated with your spark cluster.
# <br>
#


import os
from ftplib import FTP
from glob import glob
import subprocess
from pyspark.sql import Row  
from pyspark.sql.functions import regexp_replace,udf   
from pyspark.sql.types import *
from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
import pip
import requests
import lxml
import unidecode 
import pubmed_parser as pp
from azureml.logging import get_azureml_logger

run_logger = get_azureml_logger()
run_logger.log('amlrealworld.BiomedicalEntityExtraction.Download-and-Parse-XML-Spark','true')

######################################################
#   download_xml_gz_files()
######################################################
# <b>Download the files </b>
# <b> Parse the XMLs and save them as a Tab separated File </b><br>
# There are a total of 812 XML files.  It would take time for downloading that
# much data.  Its advisable to do it in batches of 50.
# Downloading and parsing 1 file takes approximately 25-30 seconds.
def download_xml_gz_files():  
    """Download MEDLINE xml files"""  
    print("Download MEDLINE xml files - start") 
    # Check current working directory.    
    print ("XML Download directory = %s" % xml_local_dir)
        
    # create the download directory if it doesn't already exist
    if not os.path.exists(xml_local_dir):
        os.makedirs(xml_local_dir)

    os.chdir(xml_local_dir)
    print ("Current working directory = %s" % os.getcwd())
        
    # connect to host, default port
    with FTP('ftp.ncbi.nlm.nih.gov', 'anonymous', '') as ftp:
        # change into "pubmed/baseline" directory
        ftp.cwd('pubmed/baseline')
        #ftp.retrlines('LIST')  
        #print (len(ftp.nlst()))
        
        # filter files by extension
        file_collection = [ x for x in ftp.nlst() if x.endswith('gz') ]        
        print(len(file_collection))
        
        for i in range(1, num_xml_files+1, batch_size):          
            file_collection = ['pubmed18n%04d.xml.gz' % j
                        for j in range(i, min([i + batch_size, num_xml_files +1]) )
                        if not os.path.exists(os.path.join(xml_local_dir,'pubmed18n%04d.xml.gz' % j))]
        
            if len(file_collection) ==0 :
                continue

            print("Downloading {} files ....".format(len(file_collection) ))

            for filename in file_collection:                

                with open(filename, 'wb') as filehandle:
                    print('\tdownloading %s .....' % filename)
                    ftp.retrbinary('RETR ' + filename, filehandle.write)
                
        ftp.quit()
    print("Download MEDLINE xml files - end\n\n") 

######################################################
#   process_files()
######################################################
def process_files():
    """Process downloaded MEDLINE folder to parquet files"""
    print("Process downloaded MEDLINE XML files - start")
    
    # Check current working directory.
   
    print ("TSV directory %s" % parse_results_remote_dir)
   
    # create the download directory if it doesn't already exist
    if not os.path.exists(xml_local_dir):
        os.makedirs(xml_local_dir)

    
    if not os.path.isdir(xml_local_dir):
        print('The directory {} does not exist'.format(xml_local_dir))
  
    for i in range(1, num_xml_files+1, batch_size):          
        file_collection = [os.path.join(xml_local_dir,'pubmed18n%04d.xml.gz' % j)
                       for j in range(i, i + batch_size) 
                       if os.path.exists(os.path.join(xml_local_dir,'pubmed18n%04d.xml.gz' % j))]        
        
        if len(file_collection) ==0 :
            continue
        
        medline_files_rdd = sc.parallelize(file_collection, numSlices=1000)      
        
        print("Processing {} files ....".format(len(file_collection) ))
        for x in file_collection:
            print('\tprocessing %s .....' % os.path.basename(x))
            dicts_out = pp.parse_medline_xml(x)
            parse_results_rdd = medline_files_rdd.\
                flatMap(lambda x: [Row(file_name=os.path.basename(x), **publication_dict)
                                for publication_dict in dicts_out])
        
        parse_results_df = parse_results_rdd.toDF()
        print("\tnumber of records before missing data removal = {}".format(parse_results_df.count()))        

        ## Drop the records where the abstracts or the titles do not have any content      
        #Remove additional new line characters present in the abstract field
        def excludeEmptyAbstracts(abstract):
            # check the number of characters and number of words in the given abstract
            if abstract is None or len(abstract) < 50 or len(abstract.split()) < 10:
                return False            
            return True 

        filterUDF = udf(excludeEmptyAbstracts, BooleanType())
        filtering_results_df = parse_results_df.\
            select("pmid","abstract").\
            withColumn("abstract", regexp_replace("abstract", "\s+", " ")).\
            where(filterUDF(parse_results_df.abstract))

        filtering_results_df.printSchema()
        print("\tnumber of records after missing data removal = {}".format(filtering_results_df.count()))

        pubmed_tsv_file = os.path.join(parse_results_remote_dir, 'batch#{}.tsv'.format(i))
        print("Writing {} records to file {} .....".format(filtering_results_df.count(), pubmed_tsv_file))        
        filtering_results_df.repartition(1).write.\
                    format("com.databricks.spark.csv").\
                    option("header", "true").\
                    option("delimiter", "\t").\
                    save(pubmed_tsv_file,  mode='overwrite')

    print("Process downloaded MEDLINE XML files - end")


sc = SparkContext.getOrCreate()
# sc = SparkContext(conf=conf)

#Specify the path of the egg file
#sc.addPyFile('wasb:///pubmed_parser-0.1-py3.5.egg')

# directory
home_dir = os.getcwd()
xml_local_dir = os.path.join(home_dir, 'medline', 'xml_files')
xml_remote_dir = os.path.join('wasb:///', 'pubmed_data', 'xml_files')
parse_results_remote_dir = os.path.join('wasb:///', 'pubmed_data', 'tsv_files')

batch_size = 50
num_xml_files = 892 
print("\t\t1_Download_and_Parse_XML_Spark.py")
print("batch_size = {}, num_xml_files = {}".format(batch_size,num_xml_files) )
download_xml_gz_files()
process_files() 

sc.stop()
