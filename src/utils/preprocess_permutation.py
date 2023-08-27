import torch


def create_permutations_cpt(label_transform,device):
    def extract_first_4(input_list):
        first_4_chars_list = [item[:4] for item in input_list]
        return first_4_chars_list

    def extract_first_3(input_list):
        first_3_chars_list = [item[:3] for item in input_list]
        return first_3_chars_list

    def extract_first_2(input_list):
        first_2_chars_list = [item[:2] for item in input_list]
        return first_2_chars_list
    def extract_first_1(input_list):
        first_1_chars_list = [item[:1] for item in input_list]
        return first_1_chars_list
    def create_mapping_dict(input_list):
        unique_strings = list(set(input_list))
        mapping_dict = {string: i for i, string in enumerate(unique_strings)}
        return mapping_dict
    def map_indices_to_integers(input_list, mapping_dict):
        mapped_indices = [mapping_dict[string] for string in input_list]
        return mapped_indices
    def create_permutation_matrix(mapped_indices, num_classes):
        num_mapped_classes = len(mapped_indices)
        permutation_matrix = torch.zeros(num_classes, num_mapped_classes)
        for i, index in enumerate(mapped_indices):
            permutation_matrix[index, i] = 1.0
        return permutation_matrix
    input_list = label_transform.get_classes()
    mapping_dict = create_mapping_dict(input_list)
    mapped_indices = map_indices_to_integers(input_list, mapping_dict)

    # Créer une matrice de permutation pour toutes les étiquettes (1991 classes)
    permutation_matrix_all = create_permutation_matrix(mapped_indices, num_classes=len(mapping_dict)).to(device)

    # Créer une matrice de permutation pour les 4 premiers caractères (nombre de classes réduit)
    input_list_4 = extract_first_4(label_transform.get_classes())
    mapping_dict_4 = create_mapping_dict(input_list_4)
    mapped_indices_4 = map_indices_to_integers(input_list_4, mapping_dict_4)
    permutation_matrix_4 = create_permutation_matrix(mapped_indices_4, num_classes=len(mapping_dict_4)).to(device)

    # Créer une matrice de permutation pour les 3 premiers caractères (nombre de classes réduit)
    input_list_3 = extract_first_3(label_transform.get_classes())
    mapping_dict_3 = create_mapping_dict(input_list_3)
    mapped_indices_3 = map_indices_to_integers(input_list_3, mapping_dict_3)
    permutation_matrix_3 = create_permutation_matrix(mapped_indices_3, num_classes=len(mapping_dict_3)).to(device)

    # Créer une matrice de permutation pour les 2 premiers caractères (nombre de classes réduit)
    input_list_2 = extract_first_2(label_transform.get_classes())
    mapping_dict_2 = create_mapping_dict(input_list_2)
    mapped_indices_2 = map_indices_to_integers(input_list_2, mapping_dict_2)
    permutation_matrix_2 = create_permutation_matrix(mapped_indices_2, num_classes=len(mapping_dict_2)).to(device)

    # Créer une matrice de permutation pour le premier caractère (nombre de classes réduit)
    input_list_1 = extract_first_1(label_transform.get_classes())
    mapping_dict_1 = create_mapping_dict(input_list_1)
    mapped_indices_1 = map_indices_to_integers(input_list_1, mapping_dict_1)
    permutation_matrix_1 = create_permutation_matrix(mapped_indices_1, num_classes=len(mapping_dict_1)).to(device)
    return permutation_matrix_1,permutation_matrix_2,permutation_matrix_3,permutation_matrix_4,permutation_matrix_all

def create_permutations_icd9(label_transform,device):


    def get_chapitre(code):
        premier_nombre = code.split('.')[0]
        if premier_nombre.isdigit():
            premier_nombre = int(premier_nombre)
            if 1 <= premier_nombre <= 139:
                return "Infectious and Parasitic Diseases (001-139)"
            elif 140 <= premier_nombre <= 239:
                return "Neoplasms (140-239)"
            elif 240 <= premier_nombre <= 279:
                return "Endocrine, Nutritional and Metabolic Diseases, and Immunity Disorders (240-279)"
            elif 280 <= premier_nombre <= 289:
                return "Diseases of the Blood and Blood-Forming Organs (280-289)"
            elif 290 <= premier_nombre <= 319:
                return "Mental Disorders (290-319)"
            elif 320 <= premier_nombre <= 389:
                return "Diseases of the Nervous System and Sense Organs (320-389)"
            elif 390 <= premier_nombre <= 459:
                return "Diseases of the Circulatory System (390-459)"
            elif 460 <= premier_nombre <= 519:
                return "Diseases of the Respiratory System (460-519)"
            elif 520 <= premier_nombre <= 579:
                return "Diseases of the Digestive System (520-579)"
            elif 580 <= premier_nombre <= 629:
                return "Diseases of the Genitourinary System (580-629)"
            elif 630 <= premier_nombre <= 679:
                return "Complications of Pregnancy, Childbirth, and the Puerperium (630-679)"
            elif 680 <= premier_nombre <= 709:
                return "Diseases of the Skin and Subcutaneous Tissue (680-709)"
            elif 710 <= premier_nombre <= 739:
                return "Diseases of the Musculoskeletal System and Connective Tissue (710-739)"
            elif 740 <= premier_nombre <= 759:
                return "Congenital Anomalies (740-759)"
            elif 760 <= premier_nombre <= 779:
                return "Certain Conditions Originating in the Perinatal Period (760-779)"
            elif 780 <= premier_nombre <= 799:
                return "Symptoms, Signs, and Ill-Defined Conditions (780-799)"
            elif 800 <= premier_nombre <= 999:
                return "Injury and Poisoning (800-999)"
        return "Chapitre inconnu"

    # Fonction pour séparer le dernier caractère après le point en deux chiffres
    def split_last_character(partie_apres_point):
        if len(partie_apres_point) == 1:
            return partie_apres_point, ''
        else:
            return partie_apres_point[0], partie_apres_point[1:]
    codes=label_transform.get_classes()

    # Liste résultante avec le chapitre, les parties avant et après le point (ou chaîne vide si pas de point)
    resultat = []
    for code in codes:
        chapitre = get_chapitre(code.split('.')[0])
        partie_avant_point = code.split('.')[0]
        partie_apres_point = code.split('.')[1] if '.' in code else ''

        if '.' in code:
            dernier_chiffre, dernier_chiffre_apres_point = split_last_character(partie_apres_point)
        else:
            dernier_chiffre, dernier_chiffre_apres_point = '', ''

        resultat.append([chapitre, partie_avant_point, dernier_chiffre, dernier_chiffre_apres_point])

    def create_mapping_dict(input_list):
        unique_strings = list(set(input_list))
        mapping_dict = {tuple_d: i for i, tuple_d in enumerate(unique_strings)}
        return mapping_dict

    def map_indices_to_integers(input_list, mapping_dict):
        mapped_indices = [mapping_dict[string] for string in input_list]
        return mapped_indices

    def create_permutation_matrix(mapped_indices, num_classes):
        num_mapped_classes = len(mapped_indices)
        permutation_matrix = torch.zeros(num_classes, num_mapped_classes)
        for i, index in enumerate(mapped_indices):
            permutation_matrix[index, i] = 1.0
        return permutation_matrix
    def tuple_list(input_list):
        ensemble_resultat = [tuple(l) for l in input_list]
        return ensemble_resultat

    def extract_level1(input_list):
        first_3_chars_list = [item[:3] for item in input_list]
        return first_3_chars_list




    def extract_section(input_list):
        first_2_chars_list = [item[:2] for item in input_list]
        return first_2_chars_list

    def extract_chapter(input_list):
        first_1_chars_list = [item[:1] for item in input_list]
        return first_1_chars_list


    input_list = tuple_list(resultat)
    mapping_dict = create_mapping_dict(input_list)
    mapped_indices = map_indices_to_integers(input_list, mapping_dict)

    permutation_matrix_all = create_permutation_matrix(mapped_indices, num_classes=len(mapping_dict)).to(device)

    # Créer une matrice de permutation pour les 4 premiers caractères (nombre de classes réduit)
    input_list_level_1 = extract_level1(tuple_list(resultat))
    mapping_dict_level_1 = create_mapping_dict(input_list_level_1)
    mapped_indices_level_1 = map_indices_to_integers(input_list_level_1, mapping_dict_level_1)
    permutation_matrix_level_1= create_permutation_matrix(mapped_indices_level_1, num_classes=len(mapping_dict_level_1)).to(device)

    # Créer une matrice de permutation pour les 3 premiers caractères (nombre de classes réduit)
    input_list_section = extract_section(tuple_list(resultat))
    mapping_dict_section = create_mapping_dict(input_list_section)
    mapped_indices_section = map_indices_to_integers(input_list_section, mapping_dict_section)
    permutation_matrix_section = create_permutation_matrix(mapped_indices_section, num_classes=len(mapping_dict_section)).to(device)

    # Créer une matrice de permutation pour les 2 premiers caractères (nombre de classes réduit)
    input_list_chapter = extract_chapter(tuple_list(resultat))
    mapping_dict_chapter = create_mapping_dict(input_list_chapter)
    mapped_indices_chapter = map_indices_to_integers(input_list_chapter, mapping_dict_chapter)
    permutation_matrix_chapter = create_permutation_matrix(mapped_indices_chapter, num_classes=len(mapping_dict_chapter)).to(device)

    return permutation_matrix_chapter,permutation_matrix_section,permutation_matrix_level_1,permutation_matrix_all

def create_permutations_icd10(label_transform,device):
    def extract_first_3(input_list):
        first_3_chars_list = [item[:3] for item in input_list]
        return first_3_chars_list
    def extract_first_2(input_list):
        first_2_chars_list = [item[:2] for item in input_list]
        return first_2_chars_list
    def extract_first_1(input_list):
        first_1_chars_list = [item[:1] for item in input_list]
        return first_1_chars_list
    def create_mapping_dict(input_list):
        unique_strings = list(set(input_list))
        mapping_dict = {string: i for i, string in enumerate(unique_strings)}
        return mapping_dict
    def map_indices_to_integers(input_list, mapping_dict):
        mapped_indices = [mapping_dict[string] for string in input_list]
        return mapped_indices
    def create_permutation_matrix(mapped_indices, num_classes):
        num_mapped_classes = len(mapped_indices)
        permutation_matrix = torch.zeros(num_classes, num_mapped_classes)
        for i, index in enumerate(mapped_indices):
            permutation_matrix[index, i] = 1.0
        return permutation_matrix
    input_list = label_transform.get_classes()
    mapping_dict = create_mapping_dict(input_list)
    mapped_indices = map_indices_to_integers(input_list, mapping_dict)

    # Créer une matrice de permutation pour toutes les étiquettes (1991 classes)
    permutation_matrix_all = create_permutation_matrix(mapped_indices, num_classes=len(mapping_dict)).to(device)

    # Créer une matrice de permutation pour les 3 premiers caractères (nombre de classes réduit)
    input_list_3 = extract_first_3(label_transform.get_classes())
    mapping_dict_3 = create_mapping_dict(input_list_3)
    mapped_indices_3 = map_indices_to_integers(input_list_3, mapping_dict_3)
    permutation_matrix_3 = create_permutation_matrix(mapped_indices_3, num_classes=len(mapping_dict_3)).to(device)

    # Créer une matrice de permutation pour les 2 premiers caractères (nombre de classes réduit)
    input_list_2 = extract_first_2(label_transform.get_classes())
    mapping_dict_2 = create_mapping_dict(input_list_2)
    mapped_indices_2 = map_indices_to_integers(input_list_2, mapping_dict_2)
    permutation_matrix_2 = create_permutation_matrix(mapped_indices_2, num_classes=len(mapping_dict_2)).to(device)

    # Créer une matrice de permutation pour le premier caractère (nombre de classes réduit)
    input_list_1 = extract_first_1(label_transform.get_classes())
    mapping_dict_1 = create_mapping_dict(input_list_1)
    mapped_indices_1 = map_indices_to_integers(input_list_1, mapping_dict_1)
    permutation_matrix_1 = create_permutation_matrix(mapped_indices_1, num_classes=len(mapping_dict_1)).to(device)
    return permutation_matrix_1,permutation_matrix_2,permutation_matrix_3,permutation_matrix_all