import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
nltk.download('punkt_tab')

nltk.download('stopwords')
stop_words = set(stopwords.words('french'))

chemins = ["resultats_finistere", "resultats_yvelines"]


corpus = []
labels = []

for label, chemin in enumerate(chemins):
    for filename in os.listdir(chemin):
        if filename.endswith('.txt'):
            filepath = os.path.join(chemin, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
            tokens = word_tokenize(text)
            filtered_tokens = [token.lower() for token in tokens if token.isalnum() and token.lower() not in stop_words]
            corpus.append(" ".join(filtered_tokens))
            labels.append(label)

vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(corpus)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_tfidf.toarray())

num_clusters = 2
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
clusters = kmeans.fit_predict(X_pca)

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.7)
plt.title('Clusterisation des textes avec KMeans')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(label='Cluster')
plt.show()

print("Clusterisation des textes :")
for i, cluster in enumerate(clusters):
    print(f"Texte {i} : Cluster {cluster}, Label réel {labels[i]}")

conf_matrix = confusion_matrix(labels, clusters)

disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["Finistère", "Yvelines"])
disp.plot(cmap='Blues', colorbar=True)
plt.title("Matrice de confusion - Clusterisation KMeans")
plt.show()

print("Clusterisation des textes :")
for i, cluster in enumerate(clusters):
    print(f"Texte {i} : Cluster {cluster}, Label réel {labels[i]}")


feature_array = np.array(vectorizer.get_feature_names_out())
keywords = defaultdict(list)

for cluster in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster)[0]

    cluster_mean_tfidf = np.mean(X_tfidf[cluster_indices].toarray(), axis=0)

    top_indices = cluster_mean_tfidf.argsort()[-10:][::-1]
    top_keywords = feature_array[top_indices]

    keywords[cluster].extend(top_keywords)

print("\nThèmes identifiés par cluster :")
for cluster, words in keywords.items():
    print(f"Cluster {cluster} : {', '.join(words)}")
