## 2.3. [Model Evaluation](4_Test_Entity_Extractor_GPU.py)

The [companion script](4_Test_Entity_Extractor_GPU.py) shows how to evaluate the Keras LSTM neural network. It uses the evaluation script from the shared task [Bio-Entity Recognition Task at Bio NLP/NLPBA 2004](http://www.nactem.ac.uk/tsujii/GENIA/ERtask/report.html) to evaluate the precision, recall, and F1 score of the model. 

### In-domain versus generic word embedding models

The following is a comparison between the accuracy of two feature types: (1) word embeddings trained on PubMed abstracts and (2) word embeddings trained on Google News. We clearly see that the in-domain model outperforms the generic model. Hence having a specific word embedding model rather than using a generic one is much more helpful. 

* Task #1: Drugs and Diseases Detection

![Model Comparison 1](../../../docs/images/mc1.png)

We perform the evaluation of the in-domain word embeddings on other datasets. The results have shown that entity extraction model trained on in-domain embeddings outperform the model trained on generic features. 

* Task #2: Proteins, Cell Line, Cell Type, DNA and RNA Detection

![Model Comparison 2](../../../docs/images/mc2.png)

* Task #3: Chemicals and Diseases Detection

![Model Comparison 3](../../../docs/images/mc3.png)

* Task #4: Drugs Detection

![Model Comparison 4](../../../docs/images/mc4.png)

* Task #5: Genes Detection

![Model Comparison 5](../../../docs/images/mc5.png)

### TensorFlow versus CNTK
All the reported models are trained using Keras with TensorFlow as backend. Keras with CNTK backend does not support "reverse" at the time this work was done. Therefore, for the sake of comparison, we have trained a unidirectional LSTM model with the CNTK backend and compared it to a unidirectional LSTM model with TensorFlow backend. Install CNTK 2.0 for Keras from [here](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras). 

![Model Comparison 6](../../../docs/images/mc6.png)

We concluded that CNTK performs as good as Tensorflow both in terms of the training time taken per epoch (60 secs for CNTK and 75 secs for Tensorflow) and the number of test entities detected. We are using the Unidirectional layers for evaluation.

### Notes

* Loading a pre-trained model for predictions

The [Python script](4_Test_Entity_Extractor_GPU.py) shows how to load a pre-trained neural entity extractor model (like the one trained in the previous step). This will be useful if you want to reuse the model 
for scoring at a later stage. The script uses the Keras load_model method. For more information about the functionalities that Keras provides
off-the-shelf for saving and loading deep learning models, refer to [here](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model).


### Next Step
3. [Deployment](./code/03_deployment/ReadMe.md)

