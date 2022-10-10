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
    
    assert result1 == ['The', 'lonely', 'wolf', 'sits', 'in', 'the', 'yellow', 'house']
    assert result2 == ['We', 'the', 'guardians', 'of', 'the', 'galaxy', 'came', 'to', 'put', 'you', 'to', 'an', 'end']
    assert result3 == ['This', 'sentence', 'contains', 'no', 'punctuation']
    assert result4 == ['Sentence', 'without', 'stopwords', 'cool', 'thing', 'guess']

def test_delete_stopwords():
    test_caption1 = ['The', 'lonely', 'wolf', 'sits', 'in', 'the', 'yellow', 'house']
    test_caption2 = ['We', 'the', 'guardians', 'of', 'the', 'galaxy', 'came', 'to', 'put', 'you', 'to', 'an', 'end']
    test_caption3 = ['This', 'sentence', 'contains', 'no', 'punctuation']
    test_caption4 = ['Sentence', 'without', 'stopwords', 'cool']

    
    result1 = datacleaner.delete_stopwords(test_caption1)
    result2 = datacleaner.delete_stopwords(test_caption2)
    result3 = datacleaner.delete_stopwords(test_caption3)
    result4 = datacleaner.delete_stopwords(test_caption4)
    
    assert result1 == ['The', 'lonely', 'wolf', 'sits', 'yellow', 'house']
    assert result2 == ['guardians', 'galaxy', 'came', 'put', 'end']
    assert result3 == ['This', 'sentence', 'contains', 'punctuation']
    assert result4 == ['Sentence', 'without', 'stopwords', 'cool']
    
    # the stopwords remover seems to be faulty, we will have to check that later

def test_remove_non_alphanumeric():
    test_caption1 = "The lonely @wolf, sits in the yellow house." 
    test_caption2 = "We, the guardians of +the galaxy, came to put you to an end !"
    test_caption3 = "This se|ntence #contains no pu|nctuation"
    test_caption4 = "Sentence without stopwords cool thing guess"
    
    result1 = datacleaner.remove_non_alphanumeric(test_caption1)
    result2 = datacleaner.remove_non_alphanumeric(test_caption2)
    result3 = datacleaner.remove_non_alphanumeric(test_caption3)
    result4 = datacleaner.remove_non_alphanumeric(test_caption4)

    assert result1 == "The lonely wolf sits in the yellow house" 
    assert result2 == "We the guardians of the galaxy came to put you to an end "
    assert result3 == "This sentence contains no punctuation"
    assert result4 == "Sentence without stopwords cool thing guess"
    

def test_lower_caption():
    test_caption1 = "The lonely Wolf, sits in The yellow house." 
    test_caption2 = "We, the guardians of The galaxy, came to put you TO an end !"
    test_caption3 = "this sentence contains no punctuation"
    test_caption4 = "SenTenCe without stopwords, cool thing guess"
    
    result1 = datacleaner.lower_caption(test_caption1)
    result2 = datacleaner.lower_caption(test_caption2)
    result3 = datacleaner.lower_caption(test_caption3)
    result4 = datacleaner.lower_caption(test_caption4)

    assert result1 == "the lonely wolf, sits in the yellow house."
    assert result2 == "we, the guardians of the galaxy, came to put you to an end !"
    assert result3 == "this sentence contains no punctuation"
    assert result4 == "sentence without stopwords, cool thing guess"
       
if __name__ == '__main__':
    test_delete_ponctuation_of_caption()    
    test_tokenize_caption()
    test_delete_stopwords()
    test_lower_caption()
    test_remove_non_alphanumeric()