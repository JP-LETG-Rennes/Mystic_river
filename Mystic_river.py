from PIL import Image
import numpy as np
from osgeo import gdal, gdalconst
import matplotlib.pyplot as plt
import pandas as pd
import os
from exif import Image as exim
from sklearn.cluster import KMeans
import datetime
import torch
from sam2.build_sam import build_sam2
from sam2.sam2image_image_predictor import SAM2ImagePredictor


def Change_format(path_dossier, format_origine, format_voulu):

    path_dossier_img = path_dossier

    lis_img = [os.path.join(path_dossier_img,e) for e in os.listdir(path_dossier_img) if e.endswith(format_origine)]

    # Il faut remplacer les extensions ".JPG" en ".jpg"

    format_voulu = format_voulu

    for e in lis_img:

        nom_original = e.split("/")[-1]

        nouveau_nom = path_dossier_img+nom_original[:-4] + f"{format_voulu}"

        os.replace(e,nouveau_nom)

def Metadata_extract(path_dossier_img, format_voulu, path_save_csv=os.getcwd(), name_csv="Resume_metadata.csv", create_csv=True):

    path_dossier_img = path_dossier_img

    format_voulu = format_voulu

    lis_img = [os.path.join(path_dossier_img, e) for e in os.listdir(path_dossier_img) if e.endswith(format_voulu)]

    dic_metadata = {"ID": [],
                    "DateTime": [],
                    "FileType": [],
                    "XResolution": [],
                    "YResolution": [],
                    "ExifImageWidth": [],
                    "ExifImageLenght": []}

    for e in lis_img:

        img = Image.open(e)

        dic_metadata['ID'] = e.split("/")[-1]

        exif_data = img._getexif()

        for i in exif_data.items():

            if i[0] == 306:

                dic_metadata["DateTime"].append(i[1])

            if i[0] == 296:

                dic_metadata["FileType"].append(i[1])

            if i[0] == 282:

                dic_metadata["XResolution"].append(i[1])

            if i[0] == 283:

                dic_metadata["YResolution"].append(i[1])

            if i[0] == 40962:

                dic_metadata["ExifImageWidth"].append(i[1])

            if i[0] == 40963:

                dic_metadata["ExifImageLenght"].append(i[1])


    if create_csv is True:

        pd.DataFrame(dic_metadata).to_csv(path_save_csv+"/"+name_csv, sep=';')

    else:

        meta = pd.DataFrame(dic_metadata)

        return meta

def img_to_csv(filepath, name_csv, filter=False):

    img = gdal.Open(filepath).ReadAsArray()

    table = None

    if filter is False:

        if len(img.shape) <= 2:

            band = img.band_Array

            ds_ravel = np.ravel(band)
            # ds_ravel = ds_ravel.reshape(-1, 1)
            dic = {'Bande1': ds_ravel}

            table = pd.DataFrame(data=dic)

        else:

            nb_bande = img.shape[2]

            if nb_bande >= 8:
                #print(img.shape)
                nb_bande = img.shape[0]

            dic = {}

            for e in range(nb_bande):

                band = img[e,:,:]

                ds_ravel = np.ravel(band)

                # ds_ravel = ds_ravel.reshape(-1, 1)

                dic['Bande{}'.format(e + 1)] = ds_ravel

            table = pd.DataFrame(data=dic)

        table.to_csv(name_csv)

        return table
    else:

        img.gaussian_filter_meth()

        if len(img.shape) <= 2:

            band = img.gaussian_filter_array

            ds_ravel = np.ravel(band)
            # ds_ravel = ds_ravel.reshape(-1, 1)

            dic = {'Bande1': ds_ravel}

            table = pd.DataFrame(data=dic)

        else:

            nb_bande = img.shape[2]

            if nb_bande >= 8:
                #print(img.shape)
                nb_bande = img.shape[0]

            dic = {}

            for e in range(nb_bande):

                band = img.gaussian_filter_array[e, :, :]

                ds_ravel = np.ravel(band)

                # ds_ravel = ds_ravel.reshape(-1, 1)

                dic['Bande{}'.format(e + 1)] = ds_ravel

            table = pd.DataFrame(data=dic)

        table.to_csv(name_csv)

        return table

def list_dos_img(path,format_voulu):
    lis_img = [os.path.join(path, e) for e in os.listdir(path) if e.endswith(format_voulu)]
    return lis_img

def Crop_MysticRiver(path, format_voulu, bbox=[0,32,2048,1056]):

    l = list_dos_img(path, format_voulu)

    for e in l:
        ds = Image.open(e)
        box = bbox
        crop = ds.crop(box)
        new_name = "Crop"+os.path.split(e)[-1]
        new_path_sav = os.path.split(e)[0]+"/"+new_name
        crop.save(new_path_sav, 'jpeg')

def Rename_date(path, format_voulu):

    l = list_dos_img(path,format_voulu)

    for e in l:
        ds = Image.open(e)

        exif = ds._getexif()

        for i in exif.items():

            if i[0] == 306:

                date = i[1]

        year, month, day, hours = date[0:4], date[5:7],date[8:10], date[11:13]

        code_rename = year+month+day+hours

        n_p = os.path.split(e)[0]+"/"

        os.rename(e, n_p+code_rename+'.jpg')

# Note de dévelopement :
"""
====================================== Organisation des données exif =================================================

Exemple: La variable 306 du dictionnaire exif est le DateTime
 Variable : Nom_metadonne
 
 306 : DateTime
 296 : FileType
 34665 : Exif_IFP_Pointer
 271 : Make
 272 : Model
 531 : ResolutionUnit
 282 : XResolution (Attention peut-être à inversé avec YResolution)
 283 : Yresolution (Attention peut-être à inversé avec XResolution)
 36864 : ExifVersion
 37121 : Je sais pas (Encodé par code propriètaire)
 40960 : Surement FlashPixVersion
 36867 : DateTimeOriginal
 36868 : DateTimeDigitized
 40961 : ColorSpace
 40962 : ExifImageWidth
 40963 : ExifImageLenght
 41986 : ExposureMode
 37385 : Flash
 41987 : WhiteBalance
 41990 : SceneCaptureType
 33434 : Je sais pas
 34855 : Surement ISOSpeedRatings
 37500 : MakerNote (Encodé par code propriètaire)


========================================== Découpage image ====================================

Les valeurs de la box pour crop est : [0,32,2048,1056]
Il y a aussi pour une découpe plus serré au niveau de la mire : [100, 600, 1750, 1000]


========================================= Gestion de la conversion pixel-distance "réel" ====================================

source :  /media/pellen_j/Transcend/Mystic-river/repereMysticRiver.ai

Sur l'image plusieurs repères :
    Sur un axe Y presque aligné (variation doit être négligeable sur axe X)
        Rocher 1 (proche) => Hauteur : 36 px ===> 40 cm donc pour 1px nous avons 1.11cm
        Rocher 2 (loin) => Hauteur : 17 px ===> 40cm donc pour 1px nous avons 2.35cm
        Mire => Hauteur : 97 px ===> 120 cm donc pour 1 px nous avons 1.24cm

Possibilité de créer matrice de distorsion des distances, peut être utile s'il n'y a pas d'eau proche de la mire.


======================================== Utilisation de SAM (Segment Anything Model) ============================================

from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
mask_generator = SamAutomaticMaskGenerator(sam)
masks = mask_generator.generate(<your_image>)



 """

if __name__ == '__main__' :

    path_dossier_test = "/home/adr2.local/pellen_j/Anne-julia/Mystic-river/RECTest/"

    path_dossier_crop = "/home/letg/Bureau/Crop/"

    path_dossier_brut = "/home/adr2.local/pellen_j/Anne-julia/Mystic-river/101RECNX/"

    format_voulu = ".jpg"



    #Change_format(path_dossier_img,".JPG", ".jpg")

    #Metadata_extract(path_dossier_img,format_voulu)

    ### Rename Image

    #Rename_date(path_dossier_brut, format_voulu)

    ### Crop Image

    #Crop_MysticRiver(path_dossier_brut, format_voulu)

    ### Crop Image plus Plus

    #Crop_MysticRiver(path_dossier_brut, format_voulu, [100, 600, 1750, 1000])

    # Test de vectorisation des bandes pour une images pour clustering par KMeans
    """
    dic_table = {}

    for e in lis_img:

        table = img_to_csv(e, name_csv="Test_Bande.csv")

        dic_table[f"{os.path.split(e)[-1]}"] = table

    orga = pd.DataFrame(dic_table)



    model = KMeans(n_clusters=5, random_state=123)

    model.fit(orga)

    dic_Kmeans = {}

    for e in lis_img:

        table = img_to_csv(e, name_csv="Test_Bande.csv")

        img_kmeans = model.predict(table)

        dic_Kmeans[f"{os.path.split(e)[-1]}"] = img_kmeans

    table_kmeans = pd.DataFrame(dic_Kmeans).to_csv("Table_Kmeans")
    """



    # Segmentation d'image par SAM


    path_img = np.array(Image.open("/home/letg/Bureau/Crop/Crop2023071011.jpg"))


    
    sam = sam_model_registry['vit_h'](checkpoint="/home/adr2.local/pellen_j/PycharmProjects/pythonProject/segment-anything/sam_vit_h_4b8939.pth")
    mask_generator = SamAutomaticMaskGenerator(sam)
    masks = mask_generator.generate(path_img)

    for dic in masks:
        for value in dic.values():
            if type(value) is type(path_img):
                # Faire le cadrage de la mire
                img = value[500:,500]
                plt.imshow(img)
                plt.show()




         


