# Persian Wikipedia Text Generation using RNN and Transformers

This project involves the development of a text generation model for Persian language text using both Recurrent Neural Networks (RNNs) and Transformer architectures. The model is trained on a Persian Wikipedia dataset, utilizing techniques like n-gram models, transfer learning, and Byte Pair Encoding (BPE) for text preprocessing.

## Project Overview

- **Goal**: To generate coherent Persian text based on a given input using models trained on Persian Wikipedia data.
- **Techniques**: RNN-based Next Token Generator, n-gram models for training, and Transformer-based models for text generation.
- **Dataset**: Persian Wikipedia dataset downloaded from Kaggle.

## Steps Involved

### 1. Data Preparation and Preprocessing

1. **Dataset Download**: The Persian Wikipedia dataset is obtained from Kaggle. Follow the steps below to download the dataset:
    ```bash
    mkdir ~/.kaggle
    cp ./kaggle.json ~/.kaggle/
    chmod 600 ~/.kaggle/kaggle.json
    kaggle datasets download miladfa7/persian-wikipedia-dataset -f Persian-WikiText-1.txt
    ```

2. **Data Cleaning**:
    - Remove noisy or unnecessary characters.
    - Remove stop words to prevent the model from overfitting on common but irrelevant words.
    - Use the `hazm` library to normalize the text, ensuring all tokens are in a consistent form.
    
    The `lemmatize` method is used to ensure words are reduced to their root form. This helps the model focus on the core meaning of the words and reduces the size of the vocabulary.

3. **Tokenization**:
    - Use **Byte Pair Encoding (BPE)** for tokenizing the data. BPE reduces the vocabulary size by merging frequent pairs of characters into a single token, improving training efficiency.
    - The benefits of BPE include handling rare words and improving the model's generalization capability. However, it might result in a larger vocabulary and more complex tokenization in some cases.

### 2. Model Training

1. **n-gram Model**:
    - The training method involves using n-grams to predict the next token in the sequence. You should investigate how n-gram models work and how they help capture patterns in the data.

2. **Model Architecture**:
    - Implement an RNN-based architecture for the Next Token Generator. You can experiment with different RNN types, including LSTM and GRU.
    - The model should be designed to generate the next token in the sequence based on the previous tokens.
    - Select hyperparameters based on dataset size and computational resources.
    - The loss function used for training should be appropriate for language modeling (typically categorical cross-entropy).

3. **Training and Visualization**:
    - Train the model and visualize the loss curve over time to monitor convergence.

### 3. Model Evaluation

- **Perplexity**: 
    - Perplexity is used to evaluate language models, representing how well the model predicts the next word. Lower perplexity indicates a better model. The formula for perplexity is:
      $$ Perplexity = 2^{H(p)} $$
      where $H(p)$ is the entropy of the model's predictions.
      
    - Use perplexity to evaluate your trained model.

### 4. Text Generation

- Use the trained model to generate text by providing a seed text as input and allowing the model to generate the subsequent tokens.
- Evaluate the fluency and coherence of the generated text.

### 5. Transformer Model for Improved Text Generation

1. **Transformer Model**:
    - Develop a Transformer-based model from scratch using the Persian Wikipedia dataset to improve the quality of generated text.
    - Use self-attention mechanisms to capture long-range dependencies and improve text generation.
    
2. **Comparison**:
    - Compare the results of the RNN-based and Transformer-based models. The Transformer model is expected to outperform the RNN model in terms of fluency and coherence.

#