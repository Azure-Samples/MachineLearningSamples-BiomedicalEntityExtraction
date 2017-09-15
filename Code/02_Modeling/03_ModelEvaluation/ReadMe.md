## **Testing the Trained Entity Extractor Model against other models**

- **Compare the performance of the model against an entity extractor trained on Google News Vectors**

Google News Vectors are trained on Google News Data. Each word has a 300-Dimensional vector. These are available [online](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit).  We wanted to compare the embeddings trained on a specific domain (Bio-Medical)
against a model trained with embeddings from a general domain like news, to evaluate if a domain-specific model achieves higher performance. Our results show that a domain-specific model indeed achieves 
higher performance. The [notebook](4_b_Test_Model_trained_on_Google_News_Embeddings.ipynb) shows the process of replicating that result.

- **Comparison between Uni-Directional LSTM trained with Pubmed Embeddings and Uni-Directional LSTM trained with Google Embeddings using Keras with CNTK backend**

The comparison between the performance of using a model with Pubmed embedding against Google News embeddings with CNTK helped us to benchmark the performance of CNTK. The [notebook 1](4_c_UniDirectional_LSTM_using_Pubmed_Embedding_with_CNTK_Backend.ipynb) 
and [notebook 2](4_d_UniDirectional_LSTM_using_Google_Embedding_with_CNTK_Backend.ipynb) demonstrates the procedure followed to implement a Uni-Directional LSTM Layer in Keras. The performance is reported in the notebooks and we see that Pubmed embeddings out-perform the general embeddings. We are using the Bio-Creative 2 [Gene Mention Identification task](http://www.biocreative.org/tasks/biocreative-i/first-task-gm/) dataset here.

Note: We are using Uni-directional LSTM layers since Keras with CNTK backend did not support "reverse" at the time this work was done.
Install CNTK 2.0 for Keras from [here](https://docs.microsoft.com/en-us/cognitive-toolkit/using-cntk-with-keras)


- **Comparison between Uni-Directional LSTM using CNTK and Tensorflow backends**

Here we are comparing the perfomance of training a model with Pubmed embeddings using Keras with CNTK and Tensorflow as the backends. The [Notebook 1](4_e_Pubmed_BC5_UniDirectional_LSTM_with_CNTK_Backend.ipynb) 
and [notebook 2](4_f_Pubmed_BC5_UniDirectional_LSTM_with_Tensorflow_Backend.ipynb) show the implementations and the results. We find that CNTK is faster than Tensorflow in terms of training time and both achieve similar over all F1 Scores 61.66 on 5343 correctly identified entities for CNTK and 60.5 on 4973 correctly identified entities for Tensorflow. We are using the Bio Creative 5 
[Disease and Chemical Identification task]( http://www.biocreative.org/tasks/biocreative-v/track-3-cdr/) dataset here.

Note:

**Loading a pre-trained model for predictions**

The [notebook](4_a_Test_the_Trained_Neural_Entity_Extractor_Model.ipynb) shows how we can load a pre-trained neural entity extractor model (like the one trained in previous section). This will be useful if you want to reuse the model for scoring at a later stage. The above notebook uses the call to load_model method by Keras. [These](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) are some useful functionalities that Keras provides off the shelf for saving and loading your deep learning models.

