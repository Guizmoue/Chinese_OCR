# Projet de Reconnaissance de Caractères Chinois

Ce projet implémente une solution de reconnaissance optique de caractères (OCR) pour les caractères chinois, en utilisant des réseaux neuronaux et des outils d'OCR open-source. Le projet est divisé en plusieurs parties : préparation des données, reconnaissance de caractères avec un modèle CRNN, comparaison des performances entre Tesseract et EasyOCR, et évaluation du module OpenVino.

## Structure du Projet

### 1. [Préparation du Corpus](pre_processing.ipynb & manuscrit.ipynb)
Les notebooks `pre_processing.ipynb` et `manuscrit.ipynb` contiennent les étapes de génération et de prétraitement du corpus d'images pour l'OCR chinois.

- **manuscrit.ipynb** : Ce notebook corrige l'inclinaison des images de texte manuscrit pour améliorer la précision de la reconnaissance. Il comprend les étapes suivantes :
  - **Chargement et affichage des images** : Utilisation de OpenCV et Matplotlib pour visualiser les images originales.
  - **Correction de l'inclinaison** : Calcul de l'angle d'inclinaison des lignes de texte dans chaque image et rotation automatique pour aligner correctement le texte.
  - **Conversion en niveaux de gris et binarisation** : Transformation des images pour éliminer le bruit et optimiser la lisibilité pour l'OCR.

  Ce prétraitement garantit que les images sont prêtes pour la reconnaissance de caractères, avec un alignement et une clarté maximisés.

- **pre_processing.ipynb** : Ce notebook complète le prétraitement en segmentant les images en lignes ou caractères individuels, et en nettoyant les images pour enlever les bruits et optimiser la qualité visuelle.

Ces étapes garantissent un corpus propre et structuré pour une utilisation optimale dans les étapes suivantes.

### 2. [CRNN OCR pour Caractères Chinois Manuscrits](CRNN_OCR_Chinese.ipynb)
Ce notebook met en œuvre un pipeline complet d'OCR pour la reconnaissance de caractères chinois manuscrits en utilisant un modèle CRNN (Convolutional Recurrent Neural Network).

#### Étapes principales :
- **Préparation des données** : Chargement et prétraitement des images de caractères chinois manuscrits.
- **Architecture du modèle** : Le modèle CRNN combine une CNN pour l'extraction des caractéristiques avec un LSTM pour la prédiction séquentielle des caractères.
- **Entraînement** : Le modèle est entraîné en optimisant sa précision de reconnaissance sur un ensemble de validation, avec visualisation des courbes d'apprentissage.
- **Évaluation** : Visualisation des performances sur des échantillons de test.
- **Réutilisation** : Test du modèle sur de nouvelles images pour évaluer sa généralisation.

#### Jeu de données :
- Origine : [Handwritten Chinese Character Datasets](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets) basé sur les bases de données de l'Institut d'Automatisation de l'Académie Chinoise des Sciences (CASIA).

### 3. [Comparaison de Tesseract OCR et EasyOCR](tesseract_vs_easyocr.ipynb)
Ce notebook compare deux outils d'OCR open-source, Tesseract et EasyOCR, en termes de précision et de performance pour la reconnaissance de caractères chinois.

#### Présentation des outils :
- **Tesseract OCR** :
  - **Avantages** : Large support multilingue, options de segmentation flexibles, algorithmes basés sur des LSTM, et personnalisation.
  - **Limites** : Précision réduite pour les caractères isolés, sensible à la qualité de l'image.
- **EasyOCR** :
  - **Avantages** : Facilité d'utilisation, support multilingue, compatible GPU.
  - **Limites** : Moins précis pour les caractères individuels, besoin de GPU pour des performances optimales.

#### Évaluation :
- Calcul des métriques de précision, rappel, F1-score et taux d'erreur pour chaque outil sur un ensemble de test.
- Génération et sauvegarde de la matrice de confusion pour visualiser les performances.

### 4. [Modèle OpenVino]()

*A COMPLETER*

## Utilisation

### Génération et Prétraitement du Corpus

1. Exécutez `manuscrit.ipynb` pour préparer et corriger les images du corpus manuscrit.

2. Exécutez `pre_processing.ipynb` pour affiner le corpus d'images.

### Entraînement du modèle CRNN
Lancez le notebook `CRNN_OCR_Chinese.ipynb` pour entraîner le modèle CRNN sur le jeu de données des caractères manuscrits chinois.

### Comparaison entre Tesseract et EasyOCR
Lancez le notebook `tesseract_vs_easyocr.ipynb` pour évaluer et comparer les performances de Tesseract et EasyOCR sur la tâche d'OCR des caractères chinois.

### Evaluation du modele OpenVino

*A COMPLETER*

## Résultats
Les résultats de chaque modèle sont sauvegardés sous forme de rapports texte et d'images, incluant les métriques de classification et la matrice de confusion des prédictions.

## Licence
Ce projet est sous licence MIT.