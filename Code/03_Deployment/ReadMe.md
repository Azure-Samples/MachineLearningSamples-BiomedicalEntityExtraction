## **Operationalize the Neural Entity Extractor Model**
In order ot operationalize a deep learning model if need to take care about installing a lot of its dependencies. For example the machine should have Python, Keras tensorflow etc. which is extremely time consuming
and error prone at the same time. In order to ensure that we do not run into any configurations related issues we can take the advantage of [Docker containers](https://blogs.msdn.microsoft.com/uk_faculty_connection/2016/09/23/getting-started-with-docker-and-container-services/).
Docker Containers essentially *Wrap a piece of software in a complete filesystem that contains everything needed to run: code, runtime, system tools, system libraries – anything that can be installed on a server. This guarantees that the software will always run the same, regardless of its environment. *
In order to make the scoring performant and real time we are using [Azure Container Service](https://docs.microsoft.com/en-us/azure/container-service/dcos-swarm/).

This set of Notebooks details the steps that will be required to Operationalize the previously trained models. This part is mostly using the tutorial for [Deploying ML Models](https://gallery.cortanaintelligence.com/Tutorial/Deploy-CNTK-model-to-ACS).
We will be using Docker Images and Azure Container Service to deploy a scoring web service that can be consumed directly. 

We also show the steps of deploying a basic website using [Flask Web app](https://docs.microsoft.com/en-us/azure/app-service-web/web-sites-python-create-deploy-flask-app) 
and create a Docker conatiner for it that can be deployed on Azure using [Web App for Linux](https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-intro). This Web app will demeonstrate how to consume the web service created through ACS.


### Section 1: [Building a Docker Image for the Web Service](5_Build_Image.ipynb)
This [Notebook](5_Build_Image.ipynb) shows how you can make a web service in Python using Flask and then make a Docker Container for it. This web service will have the trained model in its backend and will provide the model predictions for the input given. This web app will be
accessible to the outside world but will be running on your local machine. In the next sections we will illustrate how to use Azure Container Services to host the same web service without making any code changes (that's the power of Docker and Azure :) ).

Step 1: Set up the paths to where you have the trained model and other content necessary for making predictions. It is recommended to follow the previous Notebooks to obtain all the necessary files which will be 
required to set up the scoring. If you have already followed them you would have the model saved in a .model file and the content such as embeddings and data required for preparing the test set in the correct format in a 
.p Pickle file. 

Step 2: Create a file that will help you transform the input received via the web service in a format suitable for Keras. This is the place where we format the input data in the format that our pre trained model
understands. We will make use of the pickle file to load the necessary items. We also add a method (create_test_data) to help us transform the input from the web service.

Step 3: Create the Driver of the Web App. This file will contain the functions that will be called by our application. This is the place where we load the previously trained model and have a function that will help 
call this model for scoring and return the output as a JSON.

Step 4: Configure routing for the Web Service

        app.run(host='0.0.0.0') # This is the code which makes the web app accessible to the outside world.
        
Note: By Default the Flask Web App runs on port 5000. Make Sure to have this port open on your machine to allow incoming traffic. If you are using an Azure VM, create a rule in the 
[Network Security Group](https://docs.microsoft.com/en-us/azure/virtual-machines/windows/nsg-quickstart-portal)

Step 5: The following steps will help you setup a Custom Docker Image for the web app. This way you do not have to worry about the environment you are deploying to. 
The Image should conatin all the necessary items required for your web app.

- Specify the requirements in the requirements file. All these will be pip installed when you create the Docker Image.
- Create a proxy between the ports 88 and 5000. 
- Create the Docker File. This is the file that will help you install all the items listed in requirements file to your image. It is also the place where you specify the command to start the server and the port.
- Finally, you need to push this image to a Docker Repository so that it can be accessed directly using Docker on your machine or on ACS.


### Section 2: [Testing the Deployed Web Service](6_Test_Image.ipynb)
This [Notbook](6_Test_Image.ipynb) demonstrates how you can test the docker image deployed locally as well as on ACS.
It has 2 sections in it. The first section shows how to test the web service running locally on your machine. The second sections shows how to test the service once deployed on ACS (you can come back to this after finishing section 3).

The Notebook runs the Docker image created in section 1. It uses some sample text data to send to the service and outputs its response. We provide the command to run the generated docker container (via the command line). Once the docker container is up and running you will be able to send the request to the web service. If for some reason the 
service does not work you can debug it by seeing the output on the command line. Once you are satisfied that the conatiner deployed locally is exactly what you want then you can proceed to the next notebook which shows
how to deploy the same docker conatiner on ACS. 

The second section in this notebook will help you test the response from the web service deployed on ACS. You should get the same output as you got for section 1, if the service is deployed correctly.

### Section 3: [Taking this Web Service to Azure using Azure Container Service](7_Deploy_on_ACS.ipynb)
This [Notebook](7_Deploy_on_ACS.ipynb) walks through how you can deploy Azure Container Service from your notebook. 
You must have Azure CLI (for [Ubuntu](https://docs.microsoft.com/en-us/cli/azure/install-azure-cli#apt-get-for-debianubuntu)) installed on your machine for this.

**Caution**: Make sure that your model loads in less than 30 seconds. If not you may see Error 502 while accessing the service. In order to speed up the model loading, make sure you pickle the content as a dictionary 
and save all numeric items in float32 format. If you are loading multiple models for scoring this may be required.

Step 1: The first step is to setup a few details about your azure account and login.

Step 2: Next, we create a resource group for our ACS and create a SSH key that will be needed to connect to the head node later.

Step 3: Now we can create the ACS cluster based on the details provided by you in the previous steps. We also create a SSH tunnel from our machine to the ACS head node for running the deployments. 
Finally, we are ready to deploy the container we created in Section 1 onto the agents. This is done using the Marathon. Once these are done, your image is deployed on ACS are you are ready to consume it.

Step 4: You can now complete the second part of Section 2 i.e. to test your web service on ACS.

Appendix: The appendix shows how you can Tear down this created cluster if you donot need it anymore.


### Section 4: [Deploy a Website using Web App for Linux on Azure](8_Build_Website.ipynb) 
This [Notebook](8_Build_Website.ipynb) shows how you can create a flask web app, and make a docker image for it and deploy it on Azure.

Step 1: Create the UI for the web app. You can use the files in this step to modify the UI of your application. We are creating the UI files on the fly and you can edit the files in the template folder as per your
needs and have a different UI. Flask also supports accessing variables in the html documents. You can see some tutorials on its use [here](http://flask.pocoo.org/docs/0.12/tutorial/) 

Step 2: Now we create a flask web app that uses the UI created above and sends request to the web service deployed in ACS. It then used the response it got from the web service and renders it on the UI. 
This is the placewhere you specify the app url that you got after deploying the docker image on ACS. 

**Note**: This is only one of the ways to access the web service. If you want to call the web service from a different place you might need to change the code of this [notebook](5_Build_Image.ipynb) to enable CORS (for Cross Origin Requests)
You might find this [package](https://pypi.python.org/pypi/Flask-Cors) useful for doing that.

Step 3: Like any other Docker Image, now is the time to specify the requirements of your web app (packages that you would need), create a docker file to generate a custom Docker Image.

Step 4: Once the Docker Image is ready you can follow  the [Tutorial](https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-using-custom-docker-image#how-to-set-a-custom-docker-image-for-a-web-app)
for deploying the Docker Image on Azure using Web App for Linux. Remember to set the port for the web app to 5000 since that is the port we are setting in the Docker file.Have a look at 
[this](https://docs.microsoft.com/en-us/azure/app-service-web/app-service-linux-using-custom-docker-image#how-to-set-the-port-used-by-your-docker-image) for reference.

Step 5: And Violla! you have operationalized your model using the power of **Python**,  **Docker** and **Azure Container Services**. Browse to the website and see your entity extractor in action!


