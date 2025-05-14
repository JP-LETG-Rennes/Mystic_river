import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from exif import Image as exim
from osgeo import gdal
from sklearn.cluster import KMeans
import torch
from sklearn.linear_model import LinearRegression
import argparse
# SAM
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor


def change_format(path_dossier, format_origine, format_voulu):
    for f in os.listdir(path_dossier):
        if f.endswith(format_origine):
            base = os.path.splitext(f)[0]
            os.rename(os.path.join(path_dossier, f), os.path.join(path_dossier, base + format_voulu))


def list_images(path, format_voulu):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith(format_voulu)]


def metadata_extract(path_dossier_img, format_voulu, path_save_csv=os.getcwd(), name_csv="Resume_metadata.csv"):
    images = list_images(path_dossier_img, format_voulu)

    dic_metadata = {
        "ID": [], "DateTime": [], "FileType": [],
        "XResolution": [], "YResolution": [],
        "ExifImageWidth": [], "ExifImageLength": []
    }

    for path in images:
        try:
            img = Image.open(path)
            exif_data = img._getexif()
            dic_metadata['ID'].append(os.path.basename(path))
            dic_metadata["DateTime"].append(exif_data.get(306))
            dic_metadata["FileType"].append(exif_data.get(296))
            dic_metadata["XResolution"].append(exif_data.get(282))
            dic_metadata["YResolution"].append(exif_data.get(283))
            dic_metadata["ExifImageWidth"].append(exif_data.get(40962))
            dic_metadata["ExifImageLength"].append(exif_data.get(40963))
        except Exception as e:
            print(f"Erreur EXIF pour {path} : {e}")

    df = pd.DataFrame(dic_metadata)
    df.to_csv(os.path.join(path_save_csv, name_csv), sep=';')
    return df


def rename_by_date(path, format_voulu):
    for filepath in list_images(path, format_voulu):
        try:
            img = Image.open(filepath)
            exif_data = img._getexif()
            date_str = exif_data.get(306)
            if date_str:
                new_name = f"{date_str[0:4]}{date_str[5:7]}{date_str[8:10]}{date_str[11:13]}.jpg"
                os.rename(filepath, os.path.join(path, new_name))
        except Exception as e:
            print(f"Erreur renommage {filepath} : {e}")


def crop_images(path, format_voulu, bbox, suffix="Crop"):
    for filepath in list_images(path, format_voulu):
        try:
            img = Image.open(filepath)
            crop = img.crop(bbox)
            new_name = f"{suffix}_{os.path.basename(filepath)}"
            crop.save(os.path.join(path, new_name), "JPEG")
        except Exception as e:
            print(f"Erreur crop {filepath} : {e}")


def img_to_csv(filepath, name_csv, filter=False):
    img = gdal.Open(filepath).ReadAsArray()
    bands = img.shape[0] if len(img.shape) > 2 else 1
    data = {}

    for i in range(bands):
        band = img if bands == 1 else img[i, :, :]
        data[f"Bande{i + 1}"] = band.ravel()

    df = pd.DataFrame(data)
    df.to_csv(name_csv, index=False)
    return df


def load_image_as_np(path_img):
    return np.array(Image.open(path_img))

def generate_sam_masks(image_np, checkpoint_path, config_path):
    model = build_sam2(config_path, checkpoint_path)
    predictor = SAM2ImagePredictor(model)
    predictor.set_image(image_np)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        masks, _, _ = predictor.predict(image_np)
    return masks

def fuse_sam_masks(masks, image_shape):
    if len(masks) == 0:
        return np.zeros(image_shape[:2], dtype=np.uint8)

    fused = np.zeros(image_shape[:2], dtype=np.uint8)

    for i, mask in enumerate(masks):
        fused[mask > 0] = i + 1  # chaque masque a une valeur unique
    return fused

def show_mask(mask):
    plt.figure(figsize=(8, 6))
    plt.imshow(mask, cmap='jet', alpha=0.6)
    plt.colorbar()
    plt.title("Masques SAM fusionnés")
    plt.show()

def filter_images_by_white_ratio(path, format_voulu, seuil=0.1):
    """
    Retourne les images ayant une proportion de pixels blancs supérieure au seuil.

    :param path: Dossier contenant les images
    :param format_voulu: Extension des fichiers (ex: ".jpg")
    :param seuil: Seuil de proportion de blanc (ex: 0.1 = 10%)
    :return: Liste des chemins des images concernées
    """
    images_blanches = []

    for filepath in list_images(path, format_voulu):
        try:
            img = Image.open(filepath).convert('RGB')
            img_np = np.array(img)

            # Création d'un masque booléen pour pixels blancs purs
            white_pixels = np.all(img_np == [255, 255, 255], axis=-1)
            ratio_blanc = np.sum(white_pixels) / white_pixels.size

            if ratio_blanc > seuil:
                images_blanches.append(filepath)

        except Exception as e:
            print(f"Erreur analyse blanc {filepath} : {e}")

    return images_blanches

def estimer_perspective(y_pixels, tailles_reelles, afficher_graph=True):
    """
    Trouve une fonction linéaire approx. reliant Y (pixels) à une taille réelle (cm, m, etc.).

    :param y_pixels: Liste des coordonnées Y (en pixels) de repères connus.
    :param tailles_reelles: Liste des tailles réelles correspondantes (en cm, m…).
    :param afficher_graph: Affiche un graphe si True.
    :return: Coefficients (a, b) de la fonction taille = a*y + b
    """
    y_pixels = np.array(y_pixels).reshape(-1, 1)
    tailles_reelles = np.array(tailles_reelles)

    model = LinearRegression()
    model.fit(y_pixels, tailles_reelles)

    a = model.coef_[0]
    b = model.intercept_

    if afficher_graph:
        y_range = np.linspace(min(y_pixels), max(y_pixels), 100)
        prediction = model.predict(y_range.reshape(-1, 1))

        plt.scatter(y_pixels, tailles_reelles, color='red', label='Mesures')
        plt.plot(y_range, prediction, label=f'Taille ≈ {a:.3f}·Y + {b:.3f}')
        plt.xlabel('Position Y (pixels)')
        plt.ylabel('Taille réelle (cm/m)')
        plt.title('Modélisation de la perspective')
        plt.legend()
        plt.grid(True)
        plt.show()

    return a, b

def generer_image_perspective(height, width, a, b, save_path=None, show=True):
    """
    Génère une image où chaque ligne (Y) suit une valeur linéaire selon : val = a * y + b

    :param height: Hauteur de l’image en pixels.
    :param width: Largeur de l’image en pixels.
    :param a: Coefficient directeur de la fonction linéaire.
    :param b: Ordonnée à l’origine de la fonction linéaire.
    :param save_path: Chemin de sauvegarde de l’image (facultatif).
    :param show: Affiche l’image si True.
    :return: Image PIL générée.
    """
    # Créer un tableau 2D où chaque ligne Y suit la relation a * y + b
    y_coords = np.arange(height)
    valeurs = a * y_coords + b  # shape: (height,)
    valeurs = np.clip(valeurs, 0, 255)  # Clamp pour image 8 bits

    # Répliquer horizontalement pour faire une image 2D (grayscale)
    image_array = np.tile(valeurs[:, np.newaxis], (1, width)).astype(np.uint8)

    img = Image.fromarray(image_array, mode='L')  # Image en niveau de gris

    if show:
        plt.imshow(img, cmap='gray')
        plt.title(f"Image simulée avec f(y) = {a:.3f}·y + {b:.1f}")
        plt.axis('off')
        plt.show()

    if save_path:
        img.save(save_path)

    return img
# Tentative CLI


"""
def main():
    parser = argparse.ArgumentParser(description="Outils de traitement d'images - renommage, crop, filtre blanc, etc.")

    parser.add_argument('--dossier', type=str, required=True, help='Chemin du dossier contenant les images.')
    parser.add_argument('--format', type=str, default='.jpg', help='Format des fichiers image (ex: .jpg, .png).')
    parser.add_argument('--rename', action='store_true', help='Renommer les fichiers selon leur date EXIF.')
    parser.add_argument('--crop', action='store_true', help='Découper les images avec une bbox standard.')
    parser.add_argument('--crop_bbox', type=int, nargs=4, default=[0, 32, 2048, 1056],
                        help='Bounding box de crop : x1 y1 x2 y2.')
    parser.add_argument('--filtre_blanc', action='store_true',
                        help='Activer le filtre pour supprimer les images trop blanches.')
    parser.add_argument('--seuil_blanc', type=float, default=0.2, help='Seuil de proportion de blanc (ex: 0.2 = 20%).')
    parser.add_argument('--supprimer', action='store_true', help='Supprimer les images filtrées (trop de blanc).')

    args = parser.parse_args()

    if args.rename:
        print("🔄 Renommage des images par date...")
        Rename_date(args.dossier, args.format)

    if args.crop:
        print(f"✂️ Découpage des images avec bbox {args.crop_bbox}...")
        Crop_MysticRiver(args.dossier, args.format, bbox=args.crop_bbox)

    if args.filtre_blanc:
        print(f"🔍 Filtrage des images avec seuil de blanc > {args.seuil_blanc}...")
        images_a_supprimer = filter_images_by_white_ratio(args.dossier, args.format, seuil=args.seuil_blanc)
        print(f"{len(images_a_supprimer)} image(s) trouvée(s) avec trop de blanc.")

        if args.supprimer:
            for img_path in images_a_supprimer:
                os.remove(img_path)
            print("Images supprimées.")
"""

if __name__ == '__main__':
    # Paramètres
    image_path = "/home/letg/Bureau/Crop/Crop2023071011.jpg"
    checkpoint = "./checkpoints/sam2.1_hiera_large.pt"
    config = "configs/sam2.1/sam2.1_hiera_l.yaml"

    # Chargement image
    img_np = load_image_as_np(image_path)

    # Génération masques
    masks = generate_sam_masks(img_np, checkpoint, config)

    # Fusion des masques
    fused_mask = fuse_sam_masks(masks, img_np.shape)

    # Affichage
    show_mask(fused_mask)
