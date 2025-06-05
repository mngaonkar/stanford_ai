import gensim.downloader as api
from sklearn.decomposition import PCA
from gensim.models import Word2Vec, KeyedVectors
import matplotlib.pyplot as plt
import math

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))

def eculidean_norm(a: list):
    return math.sqrt(sum(x*x for x in a)) 

def cosine_similarity(a: list, b: list):
    dot_prod = dot_product(a, b)
    ecu_norm = eculidean_norm(a) * eculidean_norm(b)

    return dot_prod / ecu_norm if ecu_norm != 0 else 0

def main():
    model = api.load("glove-wiki-gigaword-100")
    print(f"type(model) = {type(model)}")
    print(f"model vector size = {model.vector_size}")
    print(f"bread vector = {model["bread"]}")

    print(f"similarity between 'bread' and 'butter' = {model.similarity('bread', 'butter')}")

    similarity = dot_product(model["bread"], model["butter"])
    print(f"similarity between 'bread' and 'butter' (dot product) = {similarity}")
    cos_sim = cosine_similarity(model["bread"], model["butter"])
    print(f"similarity between 'bread' and 'butter' (cosine similarity) = {cos_sim}")

    # find cool related words
    print(f"similar cool word to king = {model.most_similar(positive=['woman', 'king'], negative=['man'])}")

if __name__ == "__main__":
    main()



