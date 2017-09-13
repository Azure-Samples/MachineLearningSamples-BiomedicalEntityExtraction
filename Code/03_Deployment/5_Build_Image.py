
# coding: utf-8

# ### Build Docker Image that contains Entity Extractor model and Flask web application
# <br>This Notebook walks you through how to Operationalize the models we built using Docker images. We also cover how to deploy the model using Azure Container Service.<br>
# <ul>
# <li> First, we develop a Flask Web App that can be exposed to the outside world</li>
# <li> Next, we create a docker image of the Flask Web App and push it in a Docker Repo </li>
# </ul><br>
# In the next notebooks, we show how to test the web service, deploy the web service on ACS, test the deployed web service and show how to make a website to consume these created web service.
# <br><br>
# **Note**: Make sure you have docker installed on your system for testing the Docker Image later in the notebook

# In[102]:

import os
from os import path
import json


# In[ ]:

get_ipython().system(u'mkdir flaskwebapp')
get_ipython().system(u'mkdir flaskwebapp/nginx')
get_ipython().system(u'mkdir flaskwebapp/etc')
get_ipython().system(u'wget https://wcds2017summernlp.blob.core.windows.net/entityrecognition/NERmodel_D_a_D.model')
get_ipython().system(u'wget https://wcds2017summernlp.blob.core.windows.net/entityrecognition/NERmodel_D_a_D.model')


# #### Step 1<b> Copy the trained Model and the pickled content </b>

# In[ ]:

get_ipython().system(u'cp pickle_content_DDC.p flaskwebapp')
get_ipython().system(u'cp NERmodel_DDC.model flaskwebapp')
get_ipython().system(u'ls flaskwebapp')


# #### Step 2<b> Prepare the Test/Evaluation data in a format suitable for Keras </b>

# In[105]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/Data_Preparation.py', u'import numpy as np\nimport cPickle as cpickle\nfrom nltk.tokenize import sent_tokenize\nfrom nltk.tokenize import word_tokenize\n\nclass Data_Preparation:\n\n    def __init__ (self, vector_size = 100):\n        \n        # Some constants\n        self.DEFAULT_N_CLASSES = 8 #8 fro DD #12 for PBA\n        self.DEFAULT_N_FEATURES = vector_size\n        self.DEFAULT_MAX_SEQ_LENGTH = 613 #28 for PBA #208 for PBA\n        \n        # Other stuff\n        self.wordvecs = None\n        self.word_to_ix_map = {}\n        self.n_features = 0\n        self.n_tag_classes = 0\n        self.n_sentences_all = 0\n        self.tag_vector_map = {}\n        \n        self.max_sentence_len_train = 0\n        self.max_sentence_len = 0\n        \n        self.all_X_train = []\n        self.all_Y_train = []\n        \n        self.all_X_test = []\n        \n        self.read_and_parse_data()\n            \n    def get_data (self):\n        return (self.all_X_train, self.all_Y_train)\n    \n    def decode_prediction_sequence (self, pred_seq):\n        \n        pred_tags = []\n        for class_prs in pred_seq:\n            class_vec = np.zeros(self.DEFAULT_N_CLASSES, dtype=np.int32)\n            class_vec[np.argmax(class_prs)] = 1\n            if tuple(class_vec.tolist()) in self.tag_vector_map:\n                pred_tags.append(self.tag_vector_map[tuple(class_vec.tolist())])\n        return pred_tags\n    \n    def read_and_parse_data (self):\n        \n        pickle_content = cpickle.load(open("pickle_content_D_a_D.p", "rb"))        \n        self.word_to_ix_map = pickle_content["word_to_ix_map"]\n        self.wordvecs = pickle_content["wordvecs"]\n        self.DEFAULT_N_FEATURES = pickle_content["DEFAULT_N_FEATURES"]\n        self.DEFAULT_N_CLASSES = pickle_content["DEFAULT_N_CLASSES"]\n        self.max_sentence_len_train = pickle_content["max_sentence_len_train"] \n        self.tag_vector_map = pickle_content["tag_vector_map"]\n        self.zero_vec_pos = pickle_content["zero_vec_pos"]\n        return (self.all_X_train, self.all_Y_train)\n    \n    def create_test_data(self, vector_size):\n        file1 = open("test.txt")\n        abstract = ""\n        for line in file1:\n            abstract += line\n            \n        sentence_list = sent_tokenize(abstract)\n        \n        self.all_X_test = []\n        words = []\n        sentence_lengths = []\n                \n        for sentence in sentence_list:  \n            \n            elem_wordvecs = [] \n            word_list = word_tokenize(sentence)        \n            for word in word_list:\n                words.append(word)\n                w = word.lower()\n                if w in self.word_to_ix_map:\n                    elem_wordvecs.append(self.word_to_ix_map[w])\n                    \n                elif "UNK" in self.word_to_ix_map :\n                    elem_wordvecs.append(self.word_to_ix_map["UNK"])\n                \n            # Pad the sequences for missing entries to make them all the same length\n            nil_X = self.zero_vec_pos\n            pad_length = self.max_sentence_len_train - len(elem_wordvecs)\n            self.all_X_test.append( ((pad_length)*[nil_X]) + elem_wordvecs)\n            sentence_lengths.append(len(elem_wordvecs))\n        \n        self.all_X_test = np.array(self.all_X_test)\n        return self.all_X_test, words, sentence_lengths')


# #### Step 3<b> Create the driver for the Web App </b>

# In[106]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/driver.py', u'import numpy as np\nimport logging, sys, json\nimport timeit as t\nfrom keras.models import load_model\n\nfrom Data_Preparation import Data_Preparation\n\nlogger = logging.getLogger("ER_svc_logger")\nch = logging.StreamHandler(sys.stdout)\nlogger.addHandler(ch)\n\ntrainedModel = None\nmem_after_init = None\nlabelLookup = None\ntopResult = 3\n\n\ndef init():\n    """ Initialise Bi-Directional LSTM model\n    """\n    global trainedModel, labelLookup, mem_after_init, vector_size, reader\n    start = t.default_timer()\n    vector_size = 50 #Embedding Size\n    reader = Data_Preparation(vector_size)\n    \n    # Load the trained model\n    trainedModel = load_model("NERmodel_D_a_D.model")\n    \n    end = t.default_timer()\n    loadTimeMsg = "Model loading time: {0} ms".format(round((end-start)*1000, 2))\n    logger.info(loadTimeMsg)\n\n    \ndef run(content):\n    """ Classify the input using the loaded model\n    """\n    start = t.default_timer()\n    \n    ### Creating Colour Dictionary\n    colours = {}\n    colours["B-Disease"] = "blue"\n    colours["I-Disease"] = "blue"\n\n    colours["B-Drug"] = "lime"\n    colours["I-Drug"] = "lime"\n\n    colours["B-Chemical"] = "lime"\n    colours["I-Chemical"] = "lime"\n\n    colours["O"] = "black"\n\n    target = open("test.txt", "w")\n    for line in content:\n        target.write(line)\n    target.close()\n    \n    test_data, words, sentence_lengths = reader.create_test_data(vector_size)\n    \n    target = open("Pubmed_op_Output.txt", \'w\')\n    i = 0\n    # Generate Predictions for the Data from the trained model\n    for x in test_data:\n        \n        tags = trainedModel.predict(np.array([x]), batch_size=1)[0]\n        pred_tags = reader.decode_prediction_sequence(tags)\n        \n        pred_tag_wo_none = []\n        for index, tag in enumerate(pred_tags):\n            if index + sentence_lengths[i] >= len(pred_tags):\n                if tag != "NONE":\n                    pred_tag_wo_none.append(pred_tags[index])\n                else:\n                    pred_tag_wo_none.append("O")\n        \n        for wo in pred_tag_wo_none:\n            target.write(str(wo))\n            target.write("\\n")\n        target.write("\\n")\n        i+= 1\n        \n    target.close()\n    list1 = []\n    file1 = open("Pubmed_op_Output.txt")\n    for line in file1:\n        list1.append(line)\n    file1.close()\n    \n    ind = 0\n\n    #Colour Code the Text based on the Color Dictionary to identify various Entities effectively\n    text_annotated = ""\n    for word in list1:\n        w = word.split("\\n")[0]\n        if w != "":\n            if w != "O":\n                text_annotated += "<b><font size = \'2\' color = \'" + colours[w] + "\'>" + words[ind] + "</font></b> "\n            else:\n                text_annotated += "<font size = \'2\' color = \'" + colours[w] + "\'>" + words[ind] + "</font> "\n            ind += 1\n        else:\n            #Add a new line after a sentence\n            text_annotated += "<br>"\n\n    print(text_annotated)\n    end = t.default_timer()\n    logger.info("Predictions took {0} ms".format(round((end-start)*1000, 2)))\n    return (text_annotated, \'Computed in {0} ms\'.format(round((end-start)*1000, 2)))')


# #### Step 4<b> Specify the API Routes for the WebApp

# In[107]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/app.py', u'from flask import Flask, render_template, request\nfrom wtforms import Form, validators\nimport keras\nfrom driver import *\nimport time\n\napp = Flask(__name__)\n\n\n@app.route(\'/score\', methods = [\'GET\'])\ndef scoreRRS():\n    """ Endpoint for scoring\n    """\n    input = request.args.get(\'input\')\n    start = time.time()\n    response = run(input)\n    end = time.time() - start\n    dict = {}\n    dict[\'result\'] = response\n    return json.dumps(dict)\n\n\n@app.route("/")\ndef healthy():\n    return "Healthy"\n\n# Returns Keras Version\n@app.route(\'/version\', methods = [\'GET\'])\ndef version_request():\n    return keras.__version__\n\n@app.route(\'/val\', methods = [\'GET\'])\ndef val_request():\n    input = request.args.get(\'input\')\n    return input\n\nif __name__ == "__main__":\n    app.run(host=\'0.0.0.0\') # Makes the Web App accessible from the outside world\n                            # The flask web app runs on port 5000 by default. Ensure the port is open on your machine\n                            # If you are on an Azure VM, create a rule in the Network Adapter\n                            # see https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal')


# In[108]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/wsgi.py', u'import sys\nsys.path.append(\'/code/\')\nfrom app import app as application\nfrom driver import *\n\ndef create():\n    print("Initialising")\n    init()\n    application.run(host=\'127.0.0.1\', port=5000)')


# #### Step 5

# <b> List all the python requirements for your web app here. They will be pip installed in the Docker Image </b>

# In[109]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/requirements.txt', u'h5py\nwtforms\nnltk\npillow\nclick==6.7\nconfigparser==3.5.0\nFlask==0.11.1\ngunicorn==19.6.0\njson-logging-py==0.2\nMarkupSafe==1.0\nolefile==0.44\nrequests==2.12.3')


# <b>Creating a proxy between ports 88 and 5000 on Nginx Server </b>

# In[110]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/nginx/app', u'server {\n    listen 88;\n    server_name _;\n \n    location / {\n    include proxy_params;\n    proxy_pass http://127.0.0.1:5000;\n    proxy_connect_timeout 5000s;\n    proxy_read_timeout 5000s;\n  }\n}')


# Specify the name of the image as username/repository_name

# In[111]:

image_name = "akshaymehra/bidirectional_lstm_ner_ddc"
application_path = 'flaskwebapp'
docker_file_location = path.join(application_path, 'dockerfile')
print(docker_file_location)


# In[112]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/gunicorn_logging.conf', u'\n[loggers]\nkeys=root, gunicorn.error\n\n[handlers]\nkeys=console\n\n[formatters]\nkeys=json\n\n[logger_root]\nlevel=INFO\nhandlers=console\n\n[logger_gunicorn.error]\nlevel=ERROR\nhandlers=console\npropagate=0\nqualname=gunicorn.error\n\n[handler_console]\nclass=StreamHandler\nformatter=json\nargs=(sys.stdout, )\n\n[formatter_json]\nclass=jsonlogging.JSONFormatter')


# In[113]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/kill_supervisor.py', u"import sys\nimport os\nimport signal\n\n\ndef write_stdout(s):\n    sys.stdout.write(s)\n    sys.stdout.flush()\n\n# this function is modified from the code and knowledge found here: http://supervisord.org/events.html#example-event-listener-implementation\ndef main():\n    while 1:\n        write_stdout('READY\\n')\n        # wait for the event on stdin that supervisord will send\n        line = sys.stdin.readline()\n        write_stdout('Killing supervisor with this event: ' + line);\n        try:\n            # supervisord writes its pid to its file from which we read it here, see supervisord.conf\n            pidfile = open('/tmp/supervisord.pid','r')\n            pid = int(pidfile.readline());\n            os.kill(pid, signal.SIGQUIT)\n        except Exception as e:\n            write_stdout('Could not kill supervisor: ' + e.strerror + '\\n')\n            write_stdout('RESULT 2\\nOK')\n\nmain()")


# In[114]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/etc/supervisord.conf ', u'[supervisord]\nlogfile=/tmp/supervisord.log ; (main log file;default $CWD/supervisord.log)\nlogfile_maxbytes=50MB        ; (max main logfile bytes b4 rotation;default 50MB)\nlogfile_backups=10           ; (num of main logfile rotation backups;default 10)\nloglevel=info                ; (log level;default info; others: debug,warn,trace)\npidfile=/tmp/supervisord.pid ; (supervisord pidfile;default supervisord.pid)\nnodaemon=true               ; (start in foreground if true;default false)\nminfds=1024                  ; (min. avail startup file descriptors;default 1024)\nminprocs=200                 ; (min. avail process descriptors;default 200)\n\n[program:gunicorn]\ncommand=bash -c "gunicorn --workers 1 -m 007 --timeout 100000 --capture-output --error-logfile - --log-level debug --log-config gunicorn_logging.conf \\"wsgi:create()\\""\ndirectory=/code\nredirect_stderr=true\nstdout_logfile =/dev/stdout\nstdout_logfile_maxbytes=0\nstartretries=2\nstartsecs=20\n\n[program:nginx]\ncommand=/usr/sbin/nginx -g "daemon off;"\nstartretries=2\nstartsecs=5\npriority=3\n\n[eventlistener:program_exit]\ncommand=python kill_supervisor.py\ndirectory=/code\nevents=PROCESS_STATE_FATAL\npriority=2')


# <b> Creating a Custom Image with all the requirements for our web app </b>

# In[115]:

get_ipython().run_cell_magic(u'writefile', u'flaskwebapp/dockerfile', u'\nFROM ubuntu:16.04\nMAINTAINER Akshay Mehra <t-akmehr@microsoft.com>\n\nRUN mkdir /code\nWORKDIR /code\nADD . /code/\nADD etc /etc\n\nRUN apt-get update && apt-get install -y --no-install-recommends \\\n        openmpi-bin \\\n        python \\ \n        python-dev \\ \n        python-setuptools \\\n        python-pip \\\n        supervisor \\\n        nginx && \\\n    rm /etc/nginx/sites-enabled/default && \\\n    cp /code/nginx/app /etc/nginx/sites-available/ && \\\n    ln -s /etc/nginx/sites-available/app /etc/nginx/sites-enabled/ && \\\n    pip install tensorflow && \\\n    pip install keras && \\\n    pip install -r /code/requirements.txt\n\nRUN python -m nltk.downloader punkt\n\nEXPOSE 88\nCMD ["supervisord", "-c", "/etc/supervisord.conf"]')


# In[116]:

get_ipython().system(u'sudo docker build -t $image_name -f $docker_file_location $application_path --no-cache')


# In[117]:

get_ipython().system(u'sudo docker tag $image_name "docker.io/akshaymehra/bidirectional_lstm_ner_ddc"')


# In[118]:

get_ipython().system(u'sudo docker login -u akshaymehra -p Akshay2404')


# In[119]:

get_ipython().system(u'sudo docker push $image_name')


# In[120]:

print('Docker image name {}'.format(image_name)) 

