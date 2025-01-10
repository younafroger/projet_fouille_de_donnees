import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, silhouette_score
from sklearn.model_selection import train_test_split  # Ajout de l'importation manquante
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import numpy as np
from collections import defaultdict
from sklearn.svm import SVC



nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('french'))
custom_stopwords = {'tous', 'doléances', 'cette', 'leurs', 'sans', 'plus', 'les', 'nous'}
stop_words.update(custom_stopwords)

chemins = ["resultats_finistere", "resultats_yvelines"]


corpus = []
labels = []
filtered_text = []

for label, chemin in enumerate(chemins):
    for filename in os.listdir(chemin):
        if filename.endswith('.txt'):
            filepath = os.path.join(chemin, filename)
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as file:
                text = file.read()
            tokens = word_tokenize(text)
            filtered_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
            corpus.append(" ".join(filtered_tokens))
            labels.append(label)


vectorizer = TfidfVectorizer(
    max_df=0.85,
    min_df=5,
    max_features=1000,
    ngram_range=(1, 2)
)
X_tfidf = vectorizer.fit_transform(corpus)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_tfidf.toarray())

silhouette_values = []
for k in range(2, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    clusters = kmeans.fit_predict(X_pca)
    silhouette_values.append(silhouette_score(X_pca, clusters))

optimal_k = silhouette_values.index(max(silhouette_values)) + 2

print(f"Nombre optimal de clusters : {optimal_k}")

kmeans = KMeans(n_clusters=optimal_k, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Clusterisation avec KMeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.show()

feature_array = np.array(vectorizer.get_feature_names_out())
keywords = defaultdict(list)

for cluster in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster)[0]
    cluster_tfidf_sum = X_tfidf[cluster_indices].sum(axis=0)
    cluster_tfidf_mean = np.asarray(cluster_tfidf_sum).flatten() / len(cluster_indices)

    top_indices = cluster_tfidf_mean.argsort()[-10:][::-1]
    top_keywords = feature_array[top_indices]

    keywords[cluster] = top_keywords

print("\nThèmes identifiés par cluster :")
for cluster, words in keywords.items():
    print(f"Cluster {cluster} : {', '.join(words)}")

y = np.array(clusters)
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
y_pred = svm.predict(X_test)



