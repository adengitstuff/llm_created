import urllib.request
import tiktoken
from importlib.metadata import version

print("tiktoken version:", version("tiktoken"))

url = ("https://raw.githubusercontent.com/rasbt/"
 "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
 "the-verdict.txt")

file_path = "the-verdict.txt"

urllib.request.urlretrieve(url, file_path)

# now load:
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
print(f" Num characters: {len(raw_text)}")
print(raw_text[:8])

import re

# note: all regex expressions copy and pasted, the book explicitly mentions that
#   no memorization of regex syntax is necessary!

txt = "This is a text test. Let's split on whitespace"
string_whitespace_split = re.split(r'([,.]|\s)', txt)
print(txt)
print(f" after:")
print(string_whitespace_split)

txt2 = "Hello, World. -- Test!"
print(f" ==========")
print(f" txt2 : {txt2}")
result = re.split(r'([,.:;?_!"()\']|--|\s)', txt2)
print(f" raw result: ")
print(f" {result}")
result = [item.strip() for item in result if item.strip()]
print(f"")
print("")
print(f" Result 2:")
print(f" {result}")
print(f"=============" * 20)


# step 3: apply to the entire short story!
tokenized_text = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
tokenized_text = [item.strip() for item in tokenized_text if item.strip()]
print(f" Length of preprocessed:      {len(tokenized_text)}")

print(f" sample:")
print(tokenized_text[:40])

# unique token set:
all_words_in_text = sorted(set(tokenized_text)) # set removes duplicaets
total_unique_words = len(all_words_in_text)
print(f"Total unique words: {total_unique_words}")
print(f" {all_words_in_text[:55]}")

# create the set! 

vocabulary = {token:integer for integer,token in enumerate(all_words_in_text)}
for i, item in enumerate(vocabulary.items()):
    print(item)
    if i>=50: 
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {integer:string for string, integer in vocab.items()}

    def encode_text_to_token_int(self, text):
        text_split_by_spaces = re.split(r'([,.?_!"()\']|--|\s)', text)
        text_remove_whitespaces = [thing.strip() for thing in text_split_by_spaces if thing.strip()]
        ids = [self.str_to_int[s] for s in text_remove_whitespaces]
        return ids
    
    def decode_token_int_to_text(self, ids):
        text = " ".join(self.int_to_str[i] for i in ids)
        # this removes the space BEFORE punctuation:
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    

print(f" test tokenizer steps")
tokenizerx = SimpleTokenizerV1(vocabulary) # from the sample text. 
texttest = """"It's the last he painted, you know,"
            Mrs. Gisburn said with pardonable pride."""
print(f" len of texttest: {len(texttest)}")
textids = tokenizerx.encode_text_to_token_int(texttest)
print(f" len of textids: {len(textids)}")
print(textids)

print(f" ~~~~~~~~~~~~~~ Decode step:")
texts_from_ids = tokenizerx.decode_token_int_to_text(textids)
print(texts_from_ids)




print(f" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(f" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

print(f" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


all_tokens = sorted(list(set(all_words_in_text)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token:integer for integer,token in enumerate(all_tokens)}
print(len(vocab.items()))

for i, thing in enumerate(list(vocab.items())[-5:]):
    print(thing)

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}

    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
        item.strip() for item in preprocessed if item.strip()
        ]
        preprocessed = [item if item in self.str_to_int
        else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
atext1 = "Hello, do you like tea?"
atext2 = "In the sunlit terraces of the palace."
btext = " <|endoftext|> ".join((atext1, atext2))
print(btext)

#now tokenize new on vocab:
tokenizer2 = SimpleTokenizerV2(vocab)
print(f"{tokenizer2.encode(btext)}")
print(tokenizer2.decode(tokenizer2.encode(btext)))


# test tiktoken
tokenizer3 = tiktoken.get_encoding("gpt2")


print(f"=====")
text5 = (
    "Okay, this is sample text. I'm going to test the BPE in TikToken! Let's test"
)
intsfrombpe = tokenizer3.encode(text5)
print(text5)
print(intsfrombpe)
print(f"{tokenizer3.decode(intsfrombpe)}")