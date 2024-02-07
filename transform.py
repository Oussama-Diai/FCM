import numpy as np

############################################################
#   Transformations necessaires pour traiter les données
############################################################

# Fonction pour translation du nuage de points pour que les coordonnéelse:
# soient non négatives
def translation_non_negatives(data):
    # Trouver la valeur minimale pour chaque dimension
    min_values = np.min(data, axis=0)

    # Calculer la constante de translation pour chaque dimension
    translation_constants = abs(np.minimum(0,min_values))+10

    # Ajouter la constante de translation à chaque coordonnée
    translated_data = data + translation_constants

    return translated_data, translation_constants

def main():
    # Exemple d'utilisation
    # Supposons que 'your_data' est votre nuage de points sous forme de tableau NumPy
    your_data = np.array([[1, 2, 3], [-2, 5, 1], [0, 1, -4]])

    # Appliquer la translation
    translated_data, translation_constants = translation_non_negatives(your_data)

    # Afficher les résultats
    print("Données originales :\n", your_data)
    print("\nDonnées après translation :\n", translated_data)
    print("\nConstantes de translation :\n", translation_constants)

if __name__=="__main__":
    main()

