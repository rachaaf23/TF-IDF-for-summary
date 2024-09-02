import re
import os
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import T5ForConditionalGeneration, T5Tokenizer
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.bleu_score import SmoothingFunction
from rouge import Rouge

# Télécharger les données nécessaires pour nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

# Charger le modèle T5 pré-entraîné
model_t5 = T5ForConditionalGeneration.from_pretrained("t5-smal")
tokenizer_t5 = T5Tokenizer.from_pretrained("t5-small")

def format_function(data):
    if isinstance(data, str):
        # Remplacer les citations par la balise <cit>
        data = re.sub(r'\[\d+[,0-9/-]*\]', r' <cit> ', data)
        data = re.sub(r'\[\d+[" ,"/0-9/-]*\]', r' <cit> ', data)

        # Supprimer le texte entre parenthèses et crochets
        data = re.sub(r'\([^)]+\)', '', data)
        data = re.sub(r'\[.*?\]', '', data)

        # Remplacer les chiffres par la balise <dig> uniquement pour les chaînes de caractères
        data = re.sub(r'\b\d+\b', ' <dig> ', data)

        # Supprimer les tables et les figures
        data = re.sub(r'\ntable \d+.*?\n', '', data)
        data = re.sub(r'.*\t.*?\n', '', data)
        data = re.sub(r'\nfigure \d+.*?\n', '', data)
        data = re.sub(r'[(]figure \d+.*?[)]', '', data)
        data = re.sub(r'[(]fig. \d+.*?[)]', '', data)
        data = re.sub(r'[(]fig \d+.*?[)]', '', data) 

    return data

# Chemin du fichier CSV
file_path = '/kaggle/input/pfe-1-1/output1.csv'

# Lire  du DataFrame à partir du fichier CSV
data = pd.read_csv(file_path).head(15000)

# Supprimer les lignes contenant des valeurs nulles dans la colonne 'cleaned_text'
data.dropna(subset=['cleaned_text1'], inplace=True)

# Appliquer la fonction de formatage à chaque ligne de texte
data['texte_formate'] = data['cleaned_text1'].apply(format_function)

# Créer une instance de TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer()

# Liste pour stocker les résumés
résumés_1 = []

# Parcourir chaque texte dans la colonne 'texte_formate'
for index, row in data.iterrows():
    # Calculer les scores TF-IDF pour chaque phrase
    tfidf_matrix = tfidf_vectorizer.fit_transform([row['texte_formate']])
    
    # Calculer la somme des scores TF-IDF pour chaque phrase
    scores = tfidf_matrix.sum(axis=1)
    
    # Assigner les scores aux phrases
    data.at[index, 'score'] = scores[0, 0]
    
    # Tri des phrases par score de pertinence (en ordre décroissant)
    sorted_index = tfidf_matrix.toarray().argsort(axis=1)[0, ::-1]
    
    # Sélection des phrases pour le résumé
    summary_length = 15 # Nombre de phrases dans le résumé
    summary_sentences = '. '.join([sentence for i in sorted_index[:summary_length] for sentence in row['texte_formate'].split('.') if i < len(row['texte_formate'].split('.'))])
    
    # Générer le résumé abstractif à partir du résumé extractif
    inputs = tokenizer_t5.encode("summarize: " + summary_sentences, return_tensors="pt", max_length=512, truncation=True)
    outputs = model_t5.generate(inputs, max_length=150, min_length=40, length_penalty=2.0, num_beams=4, early_stopping=True)
    abstractive_summary = tokenizer_t5.decode(outputs[0], skip_special_tokens=True)
    
    # Ajouter le résumé à la liste
    résumés_1.append(abstractive_summary)

# Ajouter les résumés à la DataFrame
data['abstractive_summary'] = résumés_1

# Créer le répertoire pour stocker les données résumées s'il n'existe pas
output_folder_path = '/kaggle/working/summarized_data/'
os.makedirs(output_folder_path, exist_ok=True)

# Enregistrer tous les résumés générés dans un seul fichier CSV
output_file_path = os.path.join(output_folder_path, 'all_summaries.csv')
data[['cleaned_text1', 'abstractive_summary']].to_csv(output_file_path, index=False)

# Afficher le chemin du fichier de sortie
print("Chemin du fichier de sortie:", output_file_path)

# Charger les résumés de référence
reference_summaries = data['shorter_abstract'].tolist()

# Calculer les scores BLEU et ROUGE
bleu_scores = []
for i in range(len(résumés_1)):
    reference_summary = [reference_summaries[i].split()]
    generated_summary = résumés_1[i].split()
    bleu_score = corpus_bleu([reference_summary], [generated_summary], weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=SmoothingFunction().method1)
    bleu_scores.append(bleu_score)


rouge = Rouge()
rouge_scores = rouge.get_scores(résumés_1, reference_summaries, avg=True)

# Afficher les scores BLEU et ROUGE
print("Score BLEU moyen:", sum(bleu_scores) / len(bleu_scores))
print("Score ROUGE moyen:", rouge_scores)
