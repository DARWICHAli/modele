class Vocabulary:
    
    def __init__(self, freq_threshold):
        self.itos = {0:"<PAD>", 1:"<SOS>", 2:"<EOS>", 3:"<UNK>"}
        self.stoi = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
        self.freq = {}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
    
        print('Start building vocabulary!')

        for sentences in sentence_list: 
            for sentence in sentences:
                for word in sentence:

                    if word not in frequencies:
                        frequencies[word]=1

                    else:
                        frequencies[word] += 1

                    if frequencies[word] == self.freq_threshold:

                        self.stoi[word]  = idx
                        self.itos[idx] = word
                        idx += 1
        self.freq = frequencies
        print('Vocabulary built!')
  
    def numericalize(self, tokenized_text):

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]
