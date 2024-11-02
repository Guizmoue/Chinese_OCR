# Projet de Reconnaissance de Caractères Chinois

Ce projet implémente une solution de reconnaissance optique de caractères (OCR) pour les caractères chinois, en utilisant des réseaux neuronaux et des outils d'OCR open-source. Le projet est divisé en plusieurs parties : préparation des données, reconnaissance de caractères avec un modèle CRNN, comparaison des performances entre Tesseract et EasyOCR, et évaluation du module OpenVino.

## Structure du Projet

### 1. Préparation du Corpus
Les notebooks [`pre_processing.ipynb`](pre_processing.ipynb) et [`manuscrit.ipynb`](manuscrit.ipynb) contiennent les étapes de génération et de prétraitement du corpus d'images pour l'OCR chinois.

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

#### Évaluation :
- Calcul des métriques de précision, rappel, F1-score et taux d'erreur pour chaque outil sur un ensemble de test.
- Génération et sauvegarde de la matrice de confusion pour visualiser les performances.

### 4. [Modèle OpenVino]()

*A COMPLETER*


## Corpus utilisé (NewData120)

Le corpus **NewData120** est un sous-ensemble structuré des caractères chinois manuscrits, contenant 120 classes de caractères sélectionnés. Ce jeu de données est divisé en ensembles d'entraînement et de test.

### Description statistique

- **Distribution des classes** : Le corpus comporte 120 classes de caractères, chacune représentant un caractère chinois spécifique. Les caractères sont également répartis entre les répertoires d'entraînement et de test pour assurer une bonne représentativité.
- **Nombre total d'images** : Le corpus complet contient plusieurs milliers d'images de caractères chinois manuscrits.
- **Longueur moyenne des documents** : Chaque image contient un seul caractère, simplifiant l’analyse séquentielle et la classification.
- **Fréquence des caractères** : Les classes sont équilibrées avec une répartition similaire du nombre d'images par caractère dans chaque ensemble (entraînement et test). Une distribution des images pour chaque caractère est visualisée dans un histogramme, montrant la quantité de données disponible pour chaque classe.

### Description qualitative

- **Exemples typiques** : Chaque image contient un caractère chinois manuscrit, avec des variations dans l'épaisseur des traits, l'angle d'écriture et la clarté, reflétant les variations courantes de l'écriture humaine.
- **Difficultés linguistiques** : Certains caractères chinois se ressemblent fortement et diffèrent uniquement par quelques traits ou légères modifications structurelles. Cela crée des défis pour la différenciation automatique.
- **Caractéristiques particulières** : Le corpus inclut des images binarisées et redressées pour réduire l'impact des inclinaisons et du bruit visuel. La complexité des caractères chinois et leur structure dense exigent un modèle robuste pour distinguer efficacement les classes.


## Pourquoi ces outils et ces modèles ?

### CRNN (Convolutional Recurrent Neural Network)

#### Fonctionnalités
Le modèle **CRNN** combine des couches de réseaux de neurones convolutionnels (CNN) pour l'extraction des caractéristiques et des couches récurrentes (LSTM) pour modéliser les séquences. Il est particulièrement efficace pour des tâches de reconnaissance de caractères, notamment en raison de sa capacité à gérer des séquences de longueurs variables dans les données textuelles, comme les mots et phrases manuscrits.

#### Configuration
Le modèle est configuré pour transformer les images de texte en séquences, en extrayant des caractéristiques spatiales via le CNN puis en modélisant les relations temporelles/séquentielles via le LSTM. Ce pipeline est entraîné sur des données d'images binarisées de caractères manuscrits chinois.

#### Avantages pour la tâche
- **Robustesse pour les séquences** : Le CRNN est conçu pour traiter des images où le texte suit un ordre séquentiel, comme les caractères dans les lignes manuscrites.
- **Adaptation à l'écriture manuscrite** : En combinant CNN et LSTM, le modèle est capable de capturer les variations dans les formes des caractères, caractéristiques des écritures manuscrites.

#### Inconvénients pour la tâche
- **Besoin de ressources importantes** : L'entraînement des CRNN peut nécessiter beaucoup de mémoire et de temps de calcul, surtout pour des alphabets aussi complexes que les caractères chinois.
- **Complexité de la configuration** : La configuration et l’entraînement de ce modèle peuvent être plus complexes, nécessitant des ajustements précis de l'architecture et des hyperparamètres.

#### Justification
Le choix du CRNN repose sur sa capacité à gérer des séquences de texte complexes et variées, un atout majeur pour la reconnaissance de caractères chinois manuscrits, qui présentent souvent des variations de taille et de forme. Sa précision élevée sur les données séquentielles en fait un choix optimal pour cette tâche.


### Tesseract OCR

#### Fonctionnalités
**Tesseract OCR** est un outil open-source reconnu pour sa capacité à extraire du texte depuis des images, avec un large support multilingue et des options avancées de segmentation. Depuis sa version 4, Tesseract utilise des réseaux neuronaux (LSTM) pour améliorer la précision de la reconnaissance de texte manuscrit ou déformé.

#### Configuration
L'installation de Tesseract est simple, et il est possible de spécifier la langue et le mode de segmentation. Par exemple, `--psm 10` isole les caractères individuellement, ce qui est optimal pour les caractères chinois.

#### Avantages pour la tâche
- **Support multilingue étendu** : Tesseract prend en charge le chinois simplifié, ce qui est essentiel pour cette tâche.
- **Flexibilité de segmentation** : Les différents modes de segmentation (mot, paragraphe, caractère) permettent de s'adapter aux spécificités des images de texte manuscrit.
- **Communauté et documentation riches** : Son large support communautaire et sa documentation facilitent sa prise en main.

#### Inconvénients pour la tâche
- **Sensibilité à la qualité de l'image** : Tesseract nécessite des images de haute qualité et peut être sensible au bruit.
- **Précision limitée pour les caractères individuels** : Optimisé pour les blocs de texte, il peut être moins performant pour des caractères isolés.

#### Justification
Tesseract est un choix judicieux car il fait partie des outils les plus utilisés pour l'OCR, avec un support multilingue étendu et des options de segmentation flexibles. Il est particulièrement adapté pour extraire du texte chinois de qualité à partir d'images variées.

### EasyOCR

#### Fonctionnalités
**EasyOCR** est une bibliothèque OCR open-source, développée par Jaided AI, qui utilise des modèles de deep learning pour la reconnaissance de texte multilingue. Elle prend en charge plus de 80 langues, dont le chinois simplifié, et permet une configuration simple et rapide.

#### Configuration
EasyOCR s’installe facilement et peut utiliser un GPU pour accélérer le traitement. Elle offre également des options de segmentation, permettant de traiter les caractères individuellement.

#### Avantages pour la tâche
- **Facilité d'utilisation** : Une simple ligne de code suffit pour lancer l'OCR, et l'outil est facilement intégrable dans des workflows Python.
- **Compatibilité GPU** : Permet de traiter rapidement de grands volumes d'images.
- **Précision multilingue** : Les modèles de deep learning sont bien adaptés pour les langues complexes comme le chinois, offrant une bonne précision.

#### Inconvénients pour la tâche
- **Nécessité de GPU pour des performances optimales** : Pour des images haute résolution, un GPU est recommandé.
- **Précision variable sur les caractères individuels** : La précision peut être inégale pour des caractères isolés par rapport aux blocs de texte.

#### Justification
EasyOCR est un choix pertinent pour sa simplicité d'utilisation, sa compatibilité GPU et sa précision multilingue. Il offre une alternative efficace pour l'OCR de texte chinois, avec des performances satisfaisantes sur des images variées.


### OpenVino

#### Fonctionnalités

#### Configuration

#### Avantages pour la tâche

#### Inconvénients pour la tâche

#### Justification



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
