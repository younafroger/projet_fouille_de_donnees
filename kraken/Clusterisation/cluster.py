import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from PIL import Image

"""ce code permet de faire de la clusterisation sur nos corpus d'images, il prend en entrée deux dossier d'image et applique une clusterisation d'abord au sein des corpus
separement, puis il melange les deux dossiers d'image """

def chargement_images(dossier, image_size=(128, 128)):
    liste_image = {}
    images = []
    i = 0
    for fichier in os.listdir(dossier):
        try:
            img = img.resize(image_size)
            images.append(np.array(img).flatten())
            liste_image[i] = fichier
            i += 1
        except Exception as e:
            print(f"erreur {fichier}: {e}")
    return np.array(images), liste_image

def appliquer_pca(images1, images2, n_components=50):
    pca = PCA(n_components=n_components)
    images = np.concatenate((images1, images2), axis=0)
    pca.fit(images1)
    return pca

def appliquer_pca_m(images, n_components=50):
    pca = PCA(n_components=n_components)
    images_reduced = pca.fit_transform(images)
    return images_reduced, pca

def clusterisation_kmeans(images_pca, n_clusters=2):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(images_pca)
    return labels, kmeans

def classification_svm(images_pca, labels):
    svm = SVC(kernel='linear', random_state=42)
    svm.fit(images_pca, labels)
    return svm.predict(images_pca), svm

def afficher_clusters(images_2d, labels, titre, cmap='viridis', legend_labels=None):
    plt.figure()
    scatter = plt.scatter(images_2d[:, 0], images_2d[:, 1], c=labels, cmap=cmap)
    plt.title(titre)
    if legend_labels:
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_labels)
    plt.show()

def melanger_corpus(corpus1, corpus2):
    images = np.vstack((corpus1, corpus2))
    labels = np.array([0] * len(corpus1) + [1] * len(corpus2))
    return images, labels

def pipeline_clusterisation(dossier1, dossier2, image_size=(128, 128)):

    images1, liste_1 = chargement_images(dossier1, image_size)
    images2, liste_2 = chargement_images(dossier2, image_size)

    pca = appliquer_pca(images1, images2)
    images1_pca = pca.transform(images1)
    images2_pca = pca.transform(images2)

    labels1_kmeans, _ = clusterisation_kmeans(images1_pca)
    labels2_kmeans, _ = clusterisation_kmeans(images2_pca)

    pca = appliquer_pca(images1, images2, 2)
    images1_pca = pca.transform(images1)
    images2_pca = pca.transform(images2)

    afficher_clusters(images1_pca, labels1_kmeans, f"{dossier1} - KMeans")
    afficher_clusters(images2_pca, labels2_kmeans, f"{dossier2} - KMeans")

    labels1_svm, _ = classification_svm(images1_pca, labels1_kmeans)
    labels2_svm, _ = classification_svm(images2_pca, labels2_kmeans)

    afficher_clusters(images1_pca, labels1_svm, f"{dossier1} - SVM", cmap='coolwarm')
    afficher_clusters(images2_pca, labels2_svm, f"{dossier2} - SVM", cmap='coolwarm')

    return images1_pca, images2_pca, labels1_kmeans, labels2_kmeans

def pipeline_melange_clusterisation(dossier1, dossier2, image_size=(128, 128)):
    images1, liste_1 = chargement_images(dossier1, image_size)
    images2, liste_2 = chargement_images(dossier2, image_size)

    images_melanges, vrais_labels = melanger_corpus(images1, images2)
    images_melanges_pca, pca = appliquer_pca_m(images_melanges)

    labels_kmeans, _ = clusterisation_kmeans(images_melanges_pca)
    images_melanges_pca_2d = pca.transform(images_melanges_pca)

    afficher_clusters(images_melanges_pca_2d, labels_kmeans, "Corpus mélangé avec KMeans", legend_labels=["Finistère", "Yvelines"])

    cm = confusion_matrix(vrais_labels, labels_kmeans)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Finistère", "Yvelines"])
    disp.plot()
    plt.title("matrice de confusion avec KMeans des corpus mélangés")
    plt.show()

# Variables de chemin à modifier
dossier_finistere = "/images/Finistère"
dossier_yvelines = "/images/Yvelines"


pipeline_clusterisation(dossier_finistere, dossier_yvelines)
pipeline_melange_clusterisation(dossier_finistere, dossier_yvelines)

