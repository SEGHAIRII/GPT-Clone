from functools import lru_cache

class tokenizer:
    def __init__(self, vocab_size=10000):
        
        self.vocab_size = vocab_size
        self.encode_map = {}
        self.decode_map = {}
        
    @lru_cache(maxsize=None)
    def _to_bytes(self, text):
        """
        Convert text to bytes.
        """
        return text.encode('utf-8')
    def _find_freq_pairs(self, text):
        """
        Find the most frequent pairs in the text.
        """
        pairs = {}
        for i in range(len(text)-1):
            pair = (text[i], text[i+1])
            if pair in pairs:
                pairs[pair] += 1
            else:
                pairs[pair] = 1
        return pairs
    
    def _replace_pair(self, text, pair):
        
    
    def train(self, corpus: str, allowed_special_tokens='|<endoftext>|'):
        """train the tokenizer on the given corpus.

        Args:
            corpus (str): _description_
        """
        
        # initialize the vocab with the first 256 characters
        
        self.encode_map = {chr(i) for i in range(256)}
        
        
        # add special tokens
        self.encode_map.update({ch: 256 + i for i, ch in enumerate(allowed_special_tokens)})
        self.decode_map = {j: i for i, j in self.encode_map.items()}

        
        for new_id in range(len(self.encode_map), self.vocab_size):
            # find the most frequent pair
            pairs = self._find_freq_pairs(corpus)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            
            # add the new token to the vocab
            self.encode_map[best_pair] = new_id
            
            # update the corpus
            corpus = corpus.replace(best_pair[0] + best_pair[1], chr(new_id))
        
        
        