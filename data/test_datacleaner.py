import datacleaner

def test_delete_ponctuation_of_caption():
    test_caption1 = "The lonely wolf, sits in the yellow house." 
    test_caption2 = "We, the guardians of the galaxy, came to put you to an end !"
    test_caption3 = "This sentence contains no punctuation"
    test_caption4 = "Sentence without stopwords cool thing guess"
    
    result1 = datacleaner.delete_ponctuation_of_caption(test_caption1)
    result2 = datacleaner.delete_ponctuation_of_caption(test_caption2)
    result3 = datacleaner.delete_ponctuation_of_caption(test_caption3)
    result4 = datacleaner.delete_ponctuation_of_caption(test_caption4)

    assert result1 == "The lonely wolf sits in the yellow house"
    assert result2 == "We the guardians of the galaxy came to put you to an end "
    assert result3 == "This sentence contains no punctuation"
    assert result4 == "Sentence without stopwords cool thing guess"
    
def test_tokenize_caption():
    test_caption1 = "The lonely wolf sits in the yellow house" 
    test_caption2 = "We the guardians of the galaxy came to put you to an end"
    test_caption3 = "This sentence contains no punctuation"
    test_caption4 = "Sentence without stopwords cool thing guess"
    
    result1 = datacleaner.tokenize_caption(test_caption1)
    result2 = datacleaner.tokenize_caption(test_caption2)
    result3 = datacleaner.tokenize_caption(test_caption3)
    result4 = datacleaner.tokenize_caption(test_caption4)
    print(result1)
    assert result1 == ['The', 'lonely', 'wolf', 'sits', 'in', 'the', 'yellow', 'house']
    assert result2 == ['We', 'the', 'guardians', 'of', 'the', 'galaxy', 'came', 'to', 'put', 'you', 'to', 'an', 'end']
    assert result3 == ['This', 'sentence', 'contains', 'no', 'punctuation']
    assert result4 == ['Sentence', 'without', 'stopwords', 'cool', 'thing', 'guess']


if __name__ == '__main__':
    test_delete_ponctuation_of_caption()    
    test_tokenize_caption()