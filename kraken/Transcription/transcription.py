import os
import requests
from PIL import Image, ImageOps
import numpy as np
from skimage import filters
import kraken.binarization as blla
from kraken.lib import models
import kraken
from kraken import blla
from PIL import Image, ImageOps
from kraken.lib import vgsl, models
from kraken.rpred import rpred
import matplotlib.pyplot as plt
from skimage import filters, io
import numpy as np
import subprocess


def download_file(url, filename):
    """
    télécharge un fichier depuis une URL
    :param url: URL du fichier à télécharger
    :param filename: Chemin local pour enregistrer le fichier
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)

        print(f"fichier téléchargé {filename}")
        return filename
    except requests.exceptions.RequestException as e:
        print(f"problème dans le téléchargement du fichier : {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None

# téléchargement du modèle FoNDUE-GD
reconmodel_url = 'https://zenodo.org/records/14399779/files/FoNDUE-GD_v2_fr.mlmodel?download=1'
reconmodel_path = 'FoNDUE-GD_v2_fr.mlmodel'
if not os.path.exists(reconmodel_path):
    download_file(url=reconmodel_url, filename=reconmodel_path)

def process_image(image_path, output_path, reconmodel_path):
    """
    traite une image pour la transcription et sauvegarde le texte extrait.
    :param image_path: Chemin de l'image d'entrée.
    :param output_path: Chemin du fichier texte de sortie.
    :param reconmodel_path: Chemin du modèle de transcription.
    """
    try:
        im = Image.open(image_path)
        gray_im = ImageOps.grayscale(im)
        gray_array = np.array(gray_im)

        # binarisation
        otsu_threshold = filters.threshold_otsu(gray_array)
        binary_array = gray_array > otsu_threshold
        binary_im = Image.fromarray((binary_array * 255).astype(np.uint8))

        # segmentation et prédiction
        segresults = blla.segment(binary_im)
        reconmodel = models.load_any(reconmodel_path)
        prediction = rpred(network=reconmodel, im=binary_im, bounds=segresults)

        # sauvegarde des résultats
        text = "".join(str(record) + "\n" for record in prediction)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

        print(f"texte extrait sauvegardé dans : {output_path}")
    except Exception as e:
        print(f"erreur : {image_path}: {e}")

# chemins des dossiers
input_dir = "orig"
output_dir = "extract"

os.makedirs(output_dir, exist_ok=True)

# traitement des images
for image_name in os.listdir(input_dir):
    if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
        input_path = os.path.join(input_dir, image_name)
        output_path = os.path.join(output_dir, image_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '') + '_Fondue.txt')
        process_image(input_path, output_path, reconmodel_path)

print("traitement terminé")
