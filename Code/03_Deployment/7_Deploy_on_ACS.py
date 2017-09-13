
# coding: utf-8

# <h1>Deploy Web App on Azure Container Services (ACS)</h1>
# <br>This Notebook walks you through on how to deploy the Docker Image of the Flask web app created in earlier on the ACS
# <ul>
# <li>We show how create Azure Container Service through code and deploy our Docker image on it.</li>
# <li>We also show how to rip apart the ACS </li>
# </ul>

# ### Step 1: Setup
# Below are the various name definitions for the resources needed to setup ACS as well as the name of the Docker image we will be using.

# In[141]:

# modify these
# Avoid any special characters (like _, - etc.) in resource_group and ACS name
resource_group = "ACSResourceGroup" 
acs_name = "ACSName"
location = "South Central US"

image_name = 'akshaymehra/bidirectional_lstm_ner_ddc' 

# If you have multiple subscriptions select 
# the subscription you want to use here
selected_subscription = "'Azure Subscription'" 


# #### Azure account login
# The command below will initiate a login to your Azure account. It will pop up with an url to go to where you will enter a one off code and log into your Azure account using your browser.<br>
# In order to install Azure CLI for Ubuntu see <a href="https://docs.microsoft.com/en-us/cli/azure/install-azure-cli#apt-get-for-debianubuntu"> this </a>

# In[ ]:

get_ipython().system(u'az login -o table')


# In[142]:

get_ipython().system(u'az account set --subscription $selected_subscription')


# In[ ]:

get_ipython().system(u'az account show')


# ### Step 2: Create resources and dependencies

# #### Create resource group
# Azure encourages the use of groups to organise all the Azure components you deploy. That way it is easier to find them but also we can deleted a number of resources simply by deleting the group.

# In[ ]:

get_ipython().system(u"az group create --name $resource_group --location '$location'")


# #### Create ssh
# Create ssh key if one not present. This is needed for the tunnel we will be creating to the head node in order to interact with Marathon.

# In[145]:

import os
if not os.path.exists('{}/.ssh/id_rsa'.format(os.environ['HOME'])):
    get_ipython().system(u'ssh-keygen -t rsa -b 2048 -N "" -f ~/.ssh/id_rsa')


# ### Step 3: Deploy ACS
# We are going to deploy a small pool of 2 Standard D2 VMs. Each VM has 2 cores and 7 GB of RAM. This is the default choice when setting up an ACS cluster. 

# This step whould take roughly between 7-10 mins to execute. You can see the resources being created under the specified resource group 
# in the <a href = "portal.azure.com">Azure Portal</a>

# In[146]:

json_data = get_ipython().getoutput(u'az acs create --name $acs_name --resource-group $resource_group --admin-username mat --dns-prefix $acs_name --agent-count 2')


# In[147]:

json_dict = json.loads(''.join(json_data))


# In[ ]:

if json_dict['properties']['provisioningState'] == 'Succeeded':
    print('Succensfully provisioned ACS {}'.format(acs_name))
    _,ssh_addr,_,_,ssh_port, = json_dict['properties']['outputs']['sshMaster0']['value'].split()


# In[ ]:

get_ipython().system(u'az acs list --resource-group $resource_group --output table')


# In[ ]:

get_ipython().system(u'az acs show --name $acs_name --resource-group $resource_group')


# #### Create SSH tunnel
# Create ssh tunnel from dsvm to ACS cluster management

# In[151]:

get_ipython().run_cell_magic(u'bash', u'--bg -s $ssh_port $ssh_addr', u'ssh -o StrictHostKeyChecking=no -fNL 1212:localhost:80 -p $1 $2')


# #### Marathon deployment
# Below we create a JSON schema of our application which we will then pass to marathon. Using this schema Marathon will spin up our application in ACS.

# In[152]:

application_id = "/bidirectionallstmnerddc"


# In[153]:

app_template = {
  "id": application_id,
  "cmd": None,
  "cpus": 1,
  "mem": 1024,
  "disk": 100,
  "instances": 1,
  "acceptedResourceRoles": [
    "slave_public"
  ],
  "container": {
    "type": "DOCKER",
    "volumes": [],
    "docker": {
      "image": image_name,
      "network": "BRIDGE",
      "portMappings": [
        {
          "containerPort": 88,
          "hostPort": 80,
          "protocol": "tcp",
          "name": "80",
          "labels": {}
        }
      ],
      "privileged": False,
      "parameters": [],
      "forcePullImage": True
    }
  },
  "healthChecks": [
    {
      "path": "/",
      "protocol": "HTTP",
      "portIndex": 0,
      "gracePeriodSeconds": 300,
      "intervalSeconds": 60,
      "timeoutSeconds": 20,
      "maxConsecutiveFailures": 3
    }
  ]
}


# In[154]:

def write_json_to_file(json_dict, filename):
    with open(filename, 'w') as outfile:
        json.dump(json_dict, outfile)


# In[155]:

write_json_to_file(app_template, 'marathon.json')


# In[156]:

get_ipython().system(u'curl -X POST http://localhost:1212/marathon/v2/apps -d @marathon.json -H "Content-type: application/json"')


# In[157]:

from time import sleep
for i in range(20):
    json_data = get_ipython().getoutput(u'curl http://localhost:1212/marathon/v2/apps')
    if json.loads(json_data[-1])['apps'][0]['tasksRunning']==1:
        print('Web app ready')
        break
    else:
        print('Preparing Web app')
    sleep(10)
else:
    print('Timeout! Something went wrong!')


# In[158]:

app_url = json_dict['properties']['outputs']['agentFQDN']['value']


# In[ ]:

print('Application URL: {}'.format(app_url))
print('Application ID: {}'.format(application_id))


# ### Appendix: Tear it all down 
# Once you are done with your ACS you can use the following two commands to destroy it all.

# In[105]:

get_ipython().system(u'az acs delete --resource-group $resource_group --name $acs_name')


# In[106]:

get_ipython().system(u'az group delete --name $resource_group -y')

