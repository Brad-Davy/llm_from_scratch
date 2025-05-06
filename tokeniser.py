import re

class SimpleTokeniser:

    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {v: k for k, v in vocab.items()}

    def encode(self, text) -> list[int]:
        '''
            Takes in some text and uses seld.str_to_int to convert it to 
            a list of integers. 
        '''
        result = re.split(r"(\s+|--|[,.:;?!\"'()_|\[\]])", text)
        split_raw_data = [item.strip() for item in result if item.strip()]
        split_raw_data = [item if item in self.str_to_int else "<unk>" for item in split_raw_data]
        return [self.str_to_int[token] for token in split_raw_data]
    
    def decode(self, tokens) -> str:
        '''
            Takes in a list of integers and uses self.int_to_str to convert 
            it back to text. 
        '''
        text = " ".join([self.int_to_str[token] for token in tokens])
        text = re.sub(r"\b([A-Za-z]+)\s+'\s+([a-zA-Z])\b", r"\1'\2", text)
        text = re.sub(r"\b'\s+([a-zA-Z])", r"'\1", text)
        text = re.sub(r"\s+([,.:;?!])", r"\1", text)
        text = re.sub(r"([,.:;?!])([^\s])", r"\1 \2", text)
        text = re.sub(r"\s+", " ", text).strip()

        return text
    
if __name__ == "__main__":
    pass