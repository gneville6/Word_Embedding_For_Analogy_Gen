'''
Import some needed tools from the natural language toolkit.
These will allow us to download some dictionaries of words.
'''
import nltk.downloader
from nltk.data import find
'''
Gensim is an open-source library for unsupervised topic modeling, document indexing, retrieval by similarity, and
other natural language processing functionalities, using modern statistical machine learning.
We use it here to get the word embeddings from the 2013 word2Vec model
This model was a seminal model for word embeddings and uses a corpus of 100 billion words from the Google News
Dataset.
'''
import gensim.models

def setupModelAndGetKeyedVectors():
    """A simple function that returns a keyedvectors model for the 44k most common words in the word2vect dataset."""

    '''
    Download word2vec_sample: a sample list of the most common 44k words from the orignal google dataset.
    This step is done in case you don't have the datset locally
    '''
    nltk.download('word2vec_sample')

    # After downloading we use nltk find functionality to local where the list of words is located
    word2vec_sample = str(find('models/word2vec_sample/pruned.word2vec.txt'))

    '''
    Since we don't want to train a model from scratch we can use a pre-trained model to convert all the words in our 44k
    dataset into vectors. Here we take the word2vec_sample dataset of 44k words and convert them to there vector
    representations.
    '''
    model = gensim.models.KeyedVectors.load_word2vec_format(word2vec_sample, binary=False)
    return model


def main():
    # Get a keyed vector embedding for the 44k most common words in the google word2vect dataset
    model = setupModelAndGetKeyedVectors()

    ''' 
    Now that we have a model of all of our target words as vectors we can use some of the functionality of gensim to do
    some queries on these vectors. Here we will try the "classic" example where we take the vector representation of the
    word "king" and add to it the vector for "woman" and subtract the vector for "man"
    "king" + "woman" - "man" =
    The parameter to this call are as follows
    positive = vectors to add together
    negative = vector to subtract 
    topn = return the n closest vectors to the vector sum
    '''
    vector_arithmetic_answer = model.most_similar(positive=['woman','king'], negative=['man'], topn = 1)
    ''' 
    Print the list output of most_similar call 
    This will print the closet words and there cosine similarity to the query vector
    '''
    print(vector_arithmetic_answer)

# if your trying to run this file directly here is what you should do
if __name__ == '__main__':
    main()