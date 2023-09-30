# MAD RESPECT TO THIS DUDE
# https://medium.com/@npolovinkin/how-to-chunk-text-into-paragraphs-using-python-8ae66be38ea6

import math
from scipy.signal import argrelextrema
from sentence_transformers import util
import numpy as np


def sentencize(doc):
    sents = [sent.text for sent in doc.sents]
    return sents


def rev_sigmoid(x: float) -> float:
    return (1 / (1 + math.exp(0.5*x)))


def activate_similarities(similarities: np.array, p_size=10) -> np.array:
    """ Function returns list of weighted sums of activated sentence similarities
    Args:
        similarities (numpy array): it should square matrix where each sentence corresponds to another with cosine similarity
        p_size (int): number of sentences are used to calculate weighted sum 
    Returns:
        list: list of weighted sums
    """
    p_size = min(p_size, similarities.shape[0])

    # To create weights for sigmoid function we first have to create space. P_size will determine number of sentences used and the size of weights vector.
    x = np.linspace(-10, 10, p_size)
    # Then we need to apply activation function to the created space
    y = np.vectorize(rev_sigmoid)
    # Because we only apply activation to p_size number of sentences we have to add zeros to neglect the effect of every additional sentence and to match the length ofvector we will multiply
    activation_weights = np.pad(y(x), (0, similarities.shape[0]-p_size))
    # 1. Take each diagonal to the right of the main diagonal
    diagonals = [similarities.diagonal(each)
                 for each in range(0, similarities.shape[0])]
    # 2. Pad each diagonal by zeros at the end. Because each diagonal is different length we should pad it with zeros at the end
    diagonals = [np.pad(each, (0, similarities.shape[0]-len(each)))
                 for each in diagonals]
    # 3. Stack those diagonals into new matrix
    diagonals = np.stack(diagonals)
    # 4. Apply activation weights to each row. Multiply similarities with our activation.
    diagonals = diagonals * activation_weights.reshape(-1, 1)
    # 5. Calculate the weighted sum of activated similarities
    activated_similarities = np.sum(diagonals, axis=0)
    return activated_similarities


def get_split_points(embeddings):
    similarities = util.cos_sim(embeddings, embeddings)
    activated_similarities = activate_similarities(similarities, p_size=10)
    minmimas = argrelextrema(activated_similarities, np.less, order=1)
    split_points = {each for each in minmimas[0]}
    return split_points


def get_chunks(sents, points):
    if len(points) == 0:
        return [' '.join(sents)]

    chunks = []
    current_chunk = []
    for idx, sent in enumerate(sents):
        if idx in points:
            current_chunk.append(sent)
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        else:
            current_chunk.append(sent)

    if len(current_chunk) > 0:
        chunks.append(' '.join(current_chunk))

    return chunks


if __name__ == "__main__":
    import json
    import spacy
    import time
    from sentence_transformers import SentenceTransformer

    nlp = spacy.blank("en")
    nlp.add_pipe("sentencizer")

    MODEL = 'all-MiniLM-L6-v2'
    model = SentenceTransformer(MODEL)

    text = json.load(open('test.json'))['content']

    start = time.time()
    doc = nlp(text)
    sents = sentencize(doc)
    embeddings = model.encode(sents, batch_size=16, convert_to_numpy=True)
    split_points = get_split_points(embeddings)
    chunks = get_chunks(sents, split_points)
    print(split_points)
    print(chunks)
    print('Time elapsed: ', time.time() - start)
    print(len(chunks))
