import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = "cuda" if torch.cuda.is_available() else "cpu"
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import random
from tqdm import tqdm
import clip
import numpy as np
model, preprocess = clip.load("ViT-B/32", device=device)

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))

def get_synonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name())
    return list(set(synonyms))

def synonym_replacement(sentence, words_to_perturb):
    words = word_tokenize(sentence)
    # words = [word for word in words if word.lower() not in stop_words]
    words = [word for word in words]
    
    for _ in range(words_to_perturb):
        if not words:
            break
        word_to_replace = random.choice(words)
        synonyms = get_synonyms(word_to_replace)
        if synonyms:
            replacement = random.choice(synonyms)
            sentence = sentence.replace(word_to_replace, replacement, 1)
    
    return sentence

def random_insertion(sentence, words_to_perturb):
    words = word_tokenize(sentence)
    
    for _ in range(words_to_perturb):
        if not words:
            break
        word_to_insert = random.choice(words)
        synonyms = get_synonyms(word_to_insert)
        if synonyms:
            insertion = random.choice(synonyms)
            position = random.randint(0, len(words))
            words.insert(position, insertion)
    
    return ' '.join(words)

def random_deletion(sentence, words_to_perturb):
    words = word_tokenize(sentence)
    
    for _ in range(words_to_perturb):
        if len(words) < 2:
            break
        idx = random.randint(0, len(words)-1)
        del words[idx]
    
    return ' '.join(words)

def generate_perturbations(original_sentence, perturbation_rate, num_results):
    perturbation_types = [synonym_replacement, random_insertion, random_deletion]
    perturbed_sentences = set()
    
    for _ in tqdm(range(num_results)):
        words_to_perturb = int(len(word_tokenize(original_sentence)) * perturbation_rate)
        perturbation = random.choice(perturbation_types)
        perturbed_sentence = perturbation(original_sentence, words_to_perturb)
        perturbed_sentences.add(perturbed_sentence)
    
    return list(perturbed_sentences)

def get_data(path):
    data = {}
    i = 1
    with open(path,'r') as file:
        for line in file:
            data[i] = line.strip()
            i += 1
    return data

if __name__ == "__main__":
    df = pd.DataFrame()
    prompt = get_data('./origin_prompts/chatGPT.txt')
    for id, input in prompt.items():
        if id==6:
            result_data = []
            for rate in range(1, 5):
                prompt_2 = generate_perturbations(input, perturbation_rate=float("0." +str(rate)), num_results=100000)
                text = clip.tokenize([input]).to(device)
                result = {}
                with torch.no_grad():
                    text_features = model.encode_text(text)
                    for item in tqdm(prompt_2):
                        disturb = clip.tokenize([item]).to(device)
                        disturb_features = model.encode_text(disturb)
                        text_features = text_features.to('cpu', dtype=torch.float32)
                        disturb_features = disturb_features.to('cpu', dtype=torch.float32)
                        cosine_similarity = np.dot(text_features.flatten().numpy(), disturb_features.flatten().numpy()) / (np.linalg.norm(text_features.flatten().numpy()) * np.linalg.norm(disturb_features.flatten().numpy()))
                        result[item] = cosine_similarity

                sorted_result = dict(sorted(result.items(), key=lambda item: item[1], reverse=True))
                filtered_items = {key: value for key, value in sorted_result.items() if value > 0.85}
                print("len(filtered_items):", len(filtered_items))
                data = [len(filtered_items)] + [f"{key}: {value}" for key, value in filtered_items.items()]
                df = pd.DataFrame({f'Column {rate}': data})
                result_data.append(df)

            result_df = pd.concat(result_data, axis=1)
            out_file = f"./word_AE/result_{id}.csv"
            result_df.to_csv(out_file, index=False)
            