import string
import re
import nltk
from nltk.tokenize import word_tokenize

def delete_ponctuation_of_caption(caption):
    """
    Function that deletes punctuation of given text
    Args:
        caption (string): Sentence that needs to be cleaned of punctuation
    """
    caption_no_punctuation="".join([i for i in caption if i not in string.punctuation])
    
    return caption_no_punctuation

def lower_caption(caption):
    """
    Lowers the whole caption
    Args:
        caption (string): the caption to lower
    """
    return caption.lower()

def tokenize_caption(caption):
    """
    Caption tokenizer
    Args:
        caption (string): the caption of a picture

    Returns:
        list: a list containing the tokens
    """
    tokens = word_tokenize(caption)
    return tokens

def delete_stopwords(token_list):
    """
    Removes the stopwords from any tokenlist
    Args:
        token_list (list array): array containing the tokens

    Returns:
        list array: the same tokenlist without stopwords
    """
    stopwords = nltk.corpus.stopwords.words('english')

    output= [i for i in token_list if i not in stopwords]
    return output

def remove_non_alphanumeric(caption):
    output = ''.join([i for i in caption if i.isalnum()])
    return output

def clean_captions(dataset, column_name_of_captions):
    assert isinstance(column_name_of_captions, str)
    """
    Function removing punctuation, lowering text, removing stopwords and tokenizing the data.
    Args:
        dataset (Pandas dataframe): a pandas dataframe set
        column_name_of_captions (string): name of the column containing the captions 
    Returns:
        array = array of tokenized captions
    """
    clean_msgs= dataset[column_name_of_captions].apply(lambda x: delete_ponctuation_of_caption(x)).copy()
    clean_msgs = clean_msgs.apply(lambda x: remove_non_alphanumeric(x))
    clean_msgs_lowered = clean_msgs.apply(lambda x: lower_caption(x)).copy()
    clean_msgs_lowered_tokenized = clean_msgs_lowered.apply(lambda x: tokenize_caption(x)).copy()
    clean_msgs_lowered_tokenized_noStopwords = clean_msgs_lowered_tokenized.apply(lambda x: delete_stopwords(x))
    return clean_msgs_lowered_tokenized_noStopwords
    
    
    

    


    

