def get_test():
    import urllib.request
    url = ("https://raw.githubusercontent.com/rasbt/"
        "LLMs-from-scratch/main/ch02/01_main-chapter-code/"
        "the-verdict.txt")
    file_path = "the-verdict.txt"
    urllib.request.urlretrieve(url, file_path)



import re 

with open("the-verdict.txt", "r") as f:
    text = f.read()

print("Total number of characters in the text:", len(text))
print("First 100 characters of the text:")
print(text[:100])


def tokenise(text: str):

    result = re.split(r"(\s+|--|[,.:;?!\"'()_|\[\]])", text)
    return [item.strip() for item in result if item.strip()]

preprocessed_text = tokenise(text)
all_words = set(preprocessed_text)
vocab_size = len(all_words)
print("Vocabulary size:", vocab_size)

vocab = {token:interger for interger, token in enumerate(all_words)}
print("Vocabulary dictionary:", vocab)