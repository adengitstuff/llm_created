import urllib.request

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
result = re.split(r'([,.:;?_!"()\']|--|\s)', txt2)
result = [item.strip() for item in result if item.strip()]
print(f"")
print("")
print(f" Result 2:")
print(f" {result}")


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