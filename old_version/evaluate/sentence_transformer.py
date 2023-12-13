# import transformations, contraints, and the Augmenter
from textattack.transformations import (
    WordSwapRandomCharacterInsertion,
    WordSwapRandomCharacterDeletion,
    WordSwapRandomCharacterSubstitution,
    WordSwapNeighboringCharacterSwap,
    WordSwapQWERTY,   # 键盘错误 come from typing too quickly.  
)
from textattack.transformations import CompositeTransformation
from textattack.constraints.pre_transformation import RepeatModification
from textattack.constraints.pre_transformation import StopwordModification
from textattack.augmentation import Augmenter
# Set up transformation using CompositeTransformation()
transformation = CompositeTransformation([WordSwapRandomCharacterInsertion(),WordSwapRandomCharacterDeletion(),WordSwapQWERTY(),WordSwapRandomCharacterSubstitution(),WordSwapNeighboringCharacterSwap()])
# Set up constraints
constraints = [RepeatModification(), StopwordModification()]
from sentence_transformers import SentenceTransformer, util
import torch

embedder = SentenceTransformer('all-MiniLM-L6-v2')
augmenter = Augmenter(transformation=transformation, constraints=constraints, pct_words_to_swap=0.1, transformations_per_example=1000)

prompt_1 = "A red ball on green grass under a blue sky"
queries = [prompt_1]

corpus = augmenter.augment(prompt_1)
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

top_k = min(500, len(corpus))
for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)

    # We use cosine-similarity and torch.topk to find the highest 5 scores
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=top_k)

    print("\n\n======================\n\n")
    print("Query:", query)
    print("\nTop 5 most similar sentences in corpus:")

    for score, idx in zip(top_results[0], top_results[1]):
        print(corpus[idx], "(Score: {:.4f})".format(score))
