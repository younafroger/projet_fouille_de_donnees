import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Télécharger les ressources nécessaires pour NLTK
nltk.download('punkt')
nltk.download('stopwords')

# Initialisation
stop_words = set(stopwords.words('french'))
custom_stopwords = {'24', '25', '50' 'mars', 'vingt', 'sept', 'quatre', 'cent', 'mil', 'avril', 'deux', 'trois', 'neuf', 'mois', 'janvier', 'quils', 'quil', '22', 'leurs', 'leur', 'cette', 'ves', 're', 'ele', 'da', 'quatrevingt', 'cy', 'mars', '1789', 'cinq'}
stop_words.update(custom_stopwords)

noms = {'jean', 'yves', 'françois', 'louis', 'michel', 'joseph', 'guillaume', 'pierre', 'cosson', 'lesneven', 'alain', 'hervé', 'nicolas', 'francois', 'ollivier', 'jacques', 'marie' 'corentin', 'jeau', 'allain', 'charles', ' corentin'}
stop_words.update(noms)

# Fonction pour lire et prétraiter les fichiers d'un dossier
def read_and_process_files(folder):
    filtered_text = []
    file_names = []

    for file_name in os.listdir(folder):
        if file_name.lower().endswith('.txt'):
            with open(os.path.join(folder, file_name), 'r', encoding='utf-8') as file:
                text = file.read()

            # Tokenisation et nettoyage
            tokens = word_tokenize(text)
            filtered_tokens = [
                token.lower()
                for token in tokens
                if token.isalnum() and token.lower() not in stop_words
            ]
            filtered_text.append(" ".join(filtered_tokens))
            file_names.append(os.path.splitext(file_name)[0])

    return filtered_text, file_names

# Lecture des fichiers des deux dossiers
folders = ['transcriptions_finistere', 'transcriptions_yvelines'] #A modifier selon le corpus que l'on souhaite analyser
all_texts = []
all_file_names = []

for folder in folders:
    if os.path.exists(folder):
        texts, file_names = read_and_process_files(folder)
        all_texts.extend(texts)
        all_file_names.extend(file_names)

# Vectorisation TF-IDF
vectorizer = TfidfVectorizer(max_df=0.95, min_df=5)
X_tfidf = vectorizer.fit_transform(all_texts)

# Réduction de dimensionnalité avec PCA
pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_tfidf.toarray())

# Détermination du nombre optimal de clusters
silhouette_scores = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    silhouette_scores.append(silhouette_score(X_pca, clusters))

optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2
print(f"Nombre optimal de clusters : {optimal_k}")

# Clustering avec le nombre optimal de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

# Visualisation des clusters
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Clusterisation avec KMeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

# Extraction des mots-clés par cluster
feature_array = np.array(vectorizer.get_feature_names_out())
keywords = defaultdict(list)

for cluster in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    cluster_tfidf_sum = X_tfidf[cluster_indices].sum(axis=0)
    cluster_tfidf_mean = np.asarray(cluster_tfidf_sum).flatten() / len(cluster_indices)

    top_indices = cluster_tfidf_mean.argsort()[-10:][::-1]
    top_keywords = feature_array[top_indices]

    keywords[cluster] = top_keywords

# Affichage des mots-clés
print("\nThèmes identifiés par cluster :")
for cluster, words in keywords.items():
    print(f"Cluster {cluster} : {', '.join(words)}")

# Afficher les textes par cluster
for cluster_num in range(optimal_k):
    print(f"\nTextes du cluster {cluster_num + 1} :")
    for idx, text in enumerate(all_texts):
        if clusters[idx] == cluster_num:
            print(f"Paroisse : {all_file_names[idx]}")
