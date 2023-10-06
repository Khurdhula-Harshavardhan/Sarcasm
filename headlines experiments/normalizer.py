import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import numpy as np
from gensim.models import Word2Vec
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Normalizer():
    """
    This class acts as a module that shall normalize the dataset provided and return it! 
    This module aims to take in only text iterable data.
    """
    vectorizer = None

    def __init__(self) -> None:
        """
        The constructor call is used to set up all the attributes that shall be used by this module.
        """
        try:
            self.vectorizer = TfidfVectorizer()
        except Exception as e:
            print("[ERR] The following error occured while trying to initialize Normalizer: " +str(e))

    def clean(self, text: str) -> str:
        """
        The clean method accepts a string as a parameter, and performs basic normalization on it by:
        1. Coverting the text into either lower or upper case completely for text consistency.
        2. Extracts only the information in text and discards the rest. e.x. :puncuations, numbers, e.t.c.
        and then returns the cleaned string.
        """
        try:
            text = text.lower() #convert string into lowercase.
            text = text.replace("'","") #covert words like can't -> cant.
            text = re.findall("[a-z]+", text) #extract words.
            text = " ".join(text) #re.findall returns a list, now combine all words returned into a sentence.
            return text
        except Exception as e:
            print("[ERR] The following error occured while trying to normalize the string: "+str(e))

    def extract_text_columns(self, dataframe:pd.DataFrame) -> pd.DataFrame:
        """
        Extracts and returns a list of column names that contain text data from a pandas DataFrame.
        """
        text_columns = str()
        
        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':
                return column
        
        return None

    def cleanDataset(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        This method accepts a pandas dataframe that has text within it as datatype.
        It normalizes the text using the self.clean method and then returns the cleaned dataframe.
        """
        try:
            print("[INFO] Extracting columns containing text from dataframe.")
            column = self.extract_text_columns(dataframe=data)

            if column is None:
                raise TypeError("The dataset contains no text to be normalized!")
            
            print("[INFO] Successfully extracted text columns from the dataset.")
            print(column)

            print("[INFO] Applying Normalization over text:")
            print("[INFO]       - Converting Text into lower case for caseconsistency.")
            print("[INFO]       - Extracting only words containing alphabets.")
            data = data[column].apply(self.clean) #apply the clean method over the dataframe.
            print("[INFO] Text Normalization is now complete.")
            return data
        except TypeError as e:
            print("[ERR] The following error occured while trying to clean the dataset: "+str(e))

    def vectorize(self, data: pd.DataFrame) -> list[list]:
        """
        This method is the primary method that the user interacts with, upon calling this method by passing the dataframe containing text.
        This method fits an instance of tfidf_vectorizer over the dataset.
        Transforms it into a sparse matrix and returns it as needed by the user.
        """
        try:
            print("[INFO] Trying to create a sparse matrix for text, using an instance of TfIdf_vectorizer")
            data = self.cleanDataset(data=data) #clean the text before fitting an vectorizer over it.
            sparse_matrix = self.vectorizer.fit_transform(data) #fit and transform the data.
            print("[INFO] Fitting the vecotirzer to given text.")
            print("[INFO] Transforming the text into a sparse matrix.")

            print("[INFO] Sparse Matrix has been successfully created over the text given as input.")
            return sparse_matrix
        except Exception as e:
            print("[ERR] The following error occured while trying to create a sparse matrix: "+str(e))


    def get_matrix(self, data: pd.DataFrame) -> list[list]:
        """
        This method can be used to get a sparse matrix on text that is new, and wants to be transformed using the old vectorizer.
        """
        try:
            print("[INFO] Trying to create a sparse matrix for text, using an instance of TfIdf_vectorizer")
            data = self.cleanDataset(data=data) #clean the text before fitting an vectorizer over it.
            sparse_matrix = self.vectorizer.transform(data) #fit and transform the data.
            print("[INFO] Fitting the vecotirzer to given text.")
            print("[INFO] Transforming the text into a sparse matrix.")

            print("[INFO] Sparse Matrix has been successfully created over the text given as input.")
            return sparse_matrix
        except Exception as e:
            print("[ERR] The following error occured while trying to create a sparse matrix: "+str(e))


    def get_embeddings(self, df, max_seq_length=100, embedding_dim=110):
        """
        This method accepts the pandas dataframe and returns the embeddings resulting from it.
        """
        print("[INFO] Trying to create a word embeddings for this dataframe using Word2Vec model..")
        df = self.cleanDataset(data=df) #clean the text before fitting an vectorizer over it.
        print("[INFO] Attempting further preprocessing using Tokenizer()")

        column = self.extract_text_columns(dataframe=df)

        if column is None:
            raise TypeError("The dataset contains no text to be normalized!")
        # Preprocess the text data
        text_data = df[column].tolist()

        # Tokenize and prepare text data
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(text_data)
        sequences = tokenizer.texts_to_sequences(text_data)
        sequences = pad_sequences(sequences, maxlen=max_seq_length, padding="post", truncating="post")

        # Train a Word2Vec model
        word2vec_model = Word2Vec(sentences=[text.split() for text in text_data],
                                vector_size=embedding_dim,
                                window=5,  # Context window size
                                min_count=1,  # Minimum word frequency
                                sg=0)  # CBOW model (skip-gram: sg=1)

        # Create an embedding matrix
        embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, embedding_dim))
        for word, index in tokenizer.word_index.items():
            if word in word2vec_model.wv:
                embedding_matrix[index] = word2vec_model.wv[word]

        return embedding_matrix
