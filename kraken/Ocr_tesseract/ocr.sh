#!/bin/bash

# chemin absolu des dossiers
IMG_DIR="/home/youna/Documents/M2/OCR/Traduction"  # Dossier contenant les fichiers .jpg
OUTPUT_DIR="/home/youna/Documents/M2/OCR/results"  # Dossier où stocker les fichiers OCR

# créer le dossier de sortie s'il n'existe pas
mkdir -p "$OUTPUT_DIR"

# parcourir chaque fichier JPG dans le dossier
for img in "$IMG_DIR"/*.jpg; do
    base_name=$(basename "$img" .jpg)
    echo "Traitement de $img..."

    # ocr de l'image et sauvegarder le résultat dans un fichier .txt
    tesseract "$img" "$OUTPUT_DIR/$base_name" -l fra
done

echo "Traitement terminé"

