import re

class tokenizer:
    def __init__(self):
        self.str_to_int = {
            '<unk>': 0,
            '<pad>': 1,
            '<s>': 2,
            '</s>': 3,
        }
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}
        self.vocab_size = 4

    
    def train(self, corpus):
        """summary:
        Trains the tokenizer on the given corpus by creating a vocabulary of tokens.

        Args:   
            corpus (_type_): _description_
        """

        tokens = re.split(r'([,.:;?_!"()\"]|--|\s)', corpus)
        self.str_to_int.update({token: i + self.vocab_size for i, token in enumerate(set(tokens))})
        self.str_to_int = sorted(self.str_to_int.items(), key=lambda x: x[0])
        self.int_to_str = {v: k for k, v in self.str_to_int.items()}
        self.vocab_size += len(set(tokens))
        
    
    def encode(self, text):
        """summary:
        Encodes the given text into a list of integers using the vocabulary.

        Args:
            text (_type_): _description_

        Returns:
            _type_: _description_
        """
        tokens = re.split(r'([,.:;?_!"()\"]|--|\s)', text)
        return [self.str_to_int.get(token, self.str_to_int['<unk>']) for token in tokens]
    
    def decode(self, tokens):
        """summary:
        Decodes a list of integers back into text using the vocabulary.

        Args:
            tokens (_type_): _description_
        Returns:    
            _type_: _description_
        """
        return ''.join([self.int_to_str.get(token, '<unk>') for token in tokens])
        


        