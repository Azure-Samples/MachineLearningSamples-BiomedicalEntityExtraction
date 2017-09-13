
# coding: utf-8

# ### Create a Website to consume the Web Service created and deployed on ACS

# This Notebook provides the details of how to develop a Web App to test the Web Service hosted on ACS as created in the previous Notebook. 
# <ul>
# <li>First, create a Flask Web App that can call the ACS web service</li>
# <li>Next, create a Docker Container of the Web App</li>
# <li>Finally, deploy the Image using Web App for Linux in Azure <a href = "https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-using-custom-docker-image">Tutorial for deploying the docker Image using Webapp for Linux on Azure</a></li>
# <ul>
# 

# In[22]:

import os
from os import path
import json


# In[80]:

get_ipython().system(u'mkdir nerdetection_ddc')
get_ipython().system(u'ls nerdetection_ddc')


# #### Step 1: Create Files that will be used to render the UI

# In[ ]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/static/style.css', u'body {\n    padding-top: 5px;\n    padding-bottom: 5px;\n}\n\n/* Set padding to keep content from hitting the edges */\n.body-content {\n    padding-left: 5px;\n    padding-right: 5px;\n}\n\n/* Set width on the form input elements since they are 100% wide by default */\ninput,\nselect,\ntextarea {\n    max-width: 1000px;\n}\n\n/* styles for validation helpers */\n.field-validation-error {\n    color: #b94a48;\n}\n\n.field-validation-valid {\n    display: none;\n}\n\ninput.input-validation-error {\n    border: 1px solid #b94a48;\n}\n\ninput[type="checkbox"].input-validation-error {\n    border: 0 none;\n}\n\n.validation-summary-errors {\n    color: #b94a48;\n}\n\n.validation-summary-valid {\n    display: none;\n}')


# In[ ]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/templates/output.html', u'<!doctype html> \n<html> \n<head> \n<title>First app</title> \n<!--<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'style.css\') }}"> -->\n</head> \n<body> \n<center>\n<h1>Entity Extractor</h1> \n<form method=post action="/output">\n<textarea placeholder = "Enter your text here" name = "content" id = "content"  style="width:1000px; height:250px" >{{ val }}</textarea>\n<br>\n<input type=submit value=\'See Results\' name=\'submit_btn\'>\n<h1>Entity Extractor Output</h1> \n<p> Key: <font color="blue">Disease</font>, <font color="lime">Drug/Chemical</font>\n<form>\n<p align="left">{{ text|safe }}\n</form> \n</center>\n</body> \n</html>')


# In[ ]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/templates/first_app.html', u'<!doctype html> \n<html> \n<head> \n<title>First app</title> \n<!--<link rel="stylesheet" href="{{ url_for(\'static\', filename=\'style.css\') }}"> -->\n</head> \n<body> \n<center>\n<h1>Entity Extractor</h1> \n<form method=post action="/output">\n<textarea placeholder = "Enter your text here" name = "content" id = "content" style="width:1000px; height:250px">\nBaricitinib, Methotrexate, or Baricitinib Plus Methotrexate in Patients with Early Rheumatoid Arthritis Who Had Received Limited or No Treatment with Disease-Modifying-Anti-Rheumatic-Drugs (DMARDs): Phase 3 Trial Results.\n\nKeywords: Janus kinase (JAK), methotrexate (MTX) and rheumatoid arthritis (RA) and Clinical research.\n\nIn 2 completed phase 3 studies, baricitinib (bari) improved disease activity with a satisfactory safety profile in patients (pts) with moderately-to-severely active RA who were inadequate responders to either conventional synthetic1 or biologic2DMARDs. This abstract reports results from a phase 3 study of bari administered as monotherapy or in combination with methotrexate (MTX) to pts with early active RA who had limited or no prior treatment with DMARDs. MTX monotherapy was the active comparator.\n</textarea>\n<br>\n<input type=submit value=\'See Results\' name=\'submit_btn\'> \n</form>\n</center> \n</body> \n</html>')


# In[ ]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/templates/_formhelpers.html', u'{% macro render_field(field) %} \n\t<dt>{{ field.label }} \n\t<dd>{{ field(**kwargs)|safe }} \n\t{% if field.errors %} \n\t\t<ul class=errors> \n\t\t{% for error in field.errors %} \n\t\t\t<li>{{ error }}</li> \n\t\t{% endfor %} \n\t\t</ul> \n\t{% endif %} \n\t</dd> \n{% endmacro %}')


# #### Step 2: Create a Flask Web app that will be used to send request to the web service 

# In[81]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/app.py', u"from flask import Flask, render_template, request\nfrom wtforms import Form, validators\nimport numpy as np\nimport requests\nimport json\n\napp = Flask(__name__)\n\n@app.route('/') \ndef hello(): \n    form = request.form \n    return render_template('first_app.html', form = form) \n\n@app.route('/output', methods=['POST']) \ndef output(): \n    form = request.form\n    if request.method == 'POST':\n        content = request.form['content']\n        print content\n        app_url = 'specify the url of the service here'\n\n        scoring_url = 'http://{}/score'.format(app_url)\n\n        r = requests.get(scoring_url + '?input=' + content)\n        print(r.status_code)\n\n        string = r.content.decode('utf-8')\n        json_obj = json.loads(string)\n        text_annotated = json_obj['result'][0]   \n        print text_annotated\n        \n        return render_template('output.html', text = text_annotated, val = content)\n    \n    return render_template('first_app.html', form=form)  \n   \nif __name__ == '__main__': \n    app.run(host='0.0.0.0')")


# #### Step 3<b> Add all the requirements for your website here</b>

# In[82]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/requirements.txt', u'Flask\nwtforms\nrequests\nnumpy')


# In[83]:

image_name = "akshaymehra/nerdetection_ddc"
application_path = 'nerdetection_ddc'
docker_file_location = path.join(application_path, 'dockerfile')
print(docker_file_location)


# <b> Create a Custom Image that has all the requirements in it </b>

# In[84]:

get_ipython().run_cell_magic(u'writefile', u'nerdetection_ddc/dockerfile', u'FROM ubuntu:16.04\nMAINTAINER Akshay Mehra <t-akmehr@microsoft.com>\n\nRUN mkdir /code\nWORKDIR /code\nADD . /code/\n\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n        openmpi-bin \\\n        python \\ \n        python-dev \\ \n        python-setuptools \\\n        python-pip && \\\n        pip install -r /code/requirements.txt\n\nEXPOSE 5000\nCMD ["python", "app.py"]')


# In[85]:

get_ipython().system(u'sudo docker build -t $image_name -f $docker_file_location $application_path --no-cache')


# In[86]:

get_ipython().system(u'sudo docker login -u username -p password')


# In[87]:

get_ipython().system(u'sudo docker tag $image_name "docker.io/akshaymehra/nerdetection_ddc"')


# In[88]:

get_ipython().system(u'sudo docker push $image_name')


# In[89]:

print('Docker image name {}'.format(image_name)) 


# In[90]:

### Test Locally 
get_ipython().system(u'sudo docker run -it -p 5000:5000 akshaymehra/nerdetection_ddc')
## Access the website on port 5000 of your machine eg. http://your_machine_ip:5000/


# #### Step 4: <b> Deploy the Docker Imgae on Azure </b>

# Follow <a href = "https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-using-custom-docker-image#how-to-set-a-custom-docker-image-for-a-web-app"> this </a>to deploy your image on Azure

# Follow <a href="https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-using-custom-docker-image#how-to-set-the-port-used-by-your-docker-image"> this</a> to set the port of the web app to 5000 as specified in the Docker File
