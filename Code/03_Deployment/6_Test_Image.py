
# coding: utf-8

# ## Test web application locally
# This notebook uses some text and tests them against the web app we made previously. <br> This notebook contains 2 sections
# <ul>
# <li> The first section shows how to test the web service locally using some sample data </li>
# <li> The second section shows how to test the web service deployed on Azure Container Service </li>
# </ul>

# ### Section 1: Test the web service locally

# In[39]:

import numpy as np


# In[40]:

#specify image name here
image_name='akshaymehra/bidirectional_lstm_ner_ddc'


# In[41]:

get_ipython().system(u'sudo docker login -u akshaymehra -p Akshay2404')


# Run the Docker container in the background and open port 88

# Check the status of the service by running the below cell through the command line

# In[ ]:

#run it from the command line
#!sudo docker run -it -p 5000:88 akshaymehra/bidirectional_lstm_ner_ddc


# In[2]:

get_ipython().system(u"curl 'http://127.0.0.1:5000/version'")


# In[3]:

content = 'Insulin cures diabetes. Colon Cancer, Breast Cancer, Skin Cancer are types of Cancers'


# In[4]:

import json
import urllib.request
import requests
headers = {'content-type': 'application/json'}
json_content = json.dumps({'input':'{0}'.format(content)})
print(json_content)


# In[6]:

input = content
r = requests.get('http://127.0.0.1:5000/score?input=' + content)
print(r.status_code)


# In[7]:

string = r.content.decode('utf-8')
json_obj = json.loads(string)
val = json_obj['result'][0]
print(val)


# ### Section 2 : Test After Deployment on ACS

# In[8]:

#Specify the app URL of your ACS cluster
app_url = 'acsnameagents.southcentralus.cloudapp.azure.com'
app_id = '/bidirectionallstmnerddc'


# In[9]:

scoring_url = 'http://{}/score'.format(app_url)
version_url = 'http://{}/version'.format(app_url)


# In[10]:

get_ipython().system(u'curl $version_url # Reports the Keras Version')


# In[11]:

content = 'Insulin cures diabetes. Colon Cancer, Breast Cancer, Skin Cancer are types of Cancers'
input = content
r = requests.get(scoring_url + '?input=' + content)
print(r.status_code)


# In[12]:

string = r.content.decode('utf-8')
json_obj = json.loads(string)
val = json_obj['result'][0]


# In[13]:

print(val)

