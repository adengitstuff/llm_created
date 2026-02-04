class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for string, integer in vocab.items()}

    def encode(self, text):
        text_split_whitespaces = re.split(r'([,.?_!"()\']|--|\s)', text)