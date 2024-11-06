# Projet de Reconnaissance de Caractères Chinois

Ce projet implémente une solution de reconnaissance optique de caractères (OCR) pour les caractères chinois, en utilisant des réseaux neuronaux et des outils d'OCR open-source. Le projet est divisé en plusieurs parties : préparation des données, reconnaissance de caractères avec un modèle CRNN, comparaison des performances entre Tesseract et EasyOCR, et évaluation d'un module de l'OpenVINO pour comparer avec le modèle CRNN.

## Structure du Projet

### 1. Préparation du Corpus
Les notebooks [`pre_processing.ipynb`](pre_processing.ipynb) contient les étapes de génération et de prétraitement du corpus d'images pour l'OCR chinois.

- **pre_processing.ipynb** : Ce notebook effectue le prétraitement en segmentant les images en lignes ou caractères individuels, et en nettoyant les images pour enlever les bruits et optimiser la qualité visuelle. Ce prétraitement garantit que les images sont prêtes pour la reconnaissance de caractères, avec un alignement et une clarté maximisés.

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

### 4. Comparaison de notre modèle et [Modèle OpenVINO : handwritten-simplified-chinese-recognition-0001](chinese_handwritten_ocr.ipynb)
Ce notebook évalue le modèle [handwritten-simplified-chinese-recognition-0001]((https://github.com/OpenVINOtoolkit/open_model_zoo/blob/master/models/intel/handwritten-simplified-chinese-recognition-0001/README.md)), qui est entraîné sur du manuscrit chinois sur les deux corpus de test, les caractères manuscrits par Guilhem et [NewData120](./data/NewData120/Test/).

Nous utiliserons par la suite ses résultats de prédiction pour évaluer sa performance et comparer avec le modèle CRNN entraîné par nous.

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
L'installation de Tesseract est simple, et il est possible de spécifier la langue et le mode de segmentation. Par exemple, `--psm 10` isole les caractères individuellement, ce qui est optimal pour nous car chaque image est composée d'un seul caractère.

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


### OpenVINO : handwritten-simplified-chinese-recognition-0001

#### Fonctionnalités
OpenVINO est conçu pour rendre les modèles plus efficaces sur les matériels Intel en optimisant la structure du modèle et en réduisant le nombre de calculs requis. Cela inclut des techniques comme la quantification et la fusion de couches. Les modèles qui passent par OpenVINO sont donc adaptés pour un déploiement rapide et performant sur des processeurs Intel. Le modèle **"handwritten-simplified-chinese-recognition-0001"** est l'un des modèles configurés sur la plateforme de l'OpenVINO.


Ce modèle est conçu pour la reconnaissance de textes manuscrits en chinois simplifié. Provenant de la source PyTorch, il comprend trois composants principaux :
- un réseau neuronal convolutionnel résiduel (CNN) utilisé comme extracteur de caractéristiques
- une opération d’aplatissement
- une couche entièrement connectée servant de classificateur pour la prédiction finale.

Ce réseau est capable de reconnaître des textes en chinois simplifié composés de caractères issus de l'ensemble de données [SCUT-EPT](https://github.com/HCIILAB/SCUT-EPT_Dataset_Release).

#### Configuration
Via la plateforme de l'OpenVINO, l'installation du modèle est très facile en téléchargeant et important les modules nécessaires directement dans un notebook.
##### Input
Image en niveaux de gris, nom - `actual_input`, forme - `1, 1, 96, 2000`, format `B, C, H, W`, où :

- `B` - batch size
- `C` - nombre de canaux
- `H` - hauteur de l'image
- `W` - largeur de l'image

L'image source doit être redimensionnée à une hauteur spécifique (par exemple 96) tout en conservant le rapport hauteur/largeur, et la largeur après redimensionnement ne doit pas être supérieure à 2000, puis la largeur doit être complétée en bas à droite jusqu'à 2000 avec des valeurs de bord.

##### Outputs
Nom - `output`, forme - `125, 1, 4059`, format `W, B, L`, où :

- `W` - longueur de la séquence de sortie
- `B` - batch size
- `L` - distribution de confiance sur les symboles pris en charge par le modèle


#### Avantages pour la tâche
- **Modèle léger** : La prédiction effectuée par le modèle ne demande pas de GPU.
- **Adaptabilité** : OpenVINO permet d’adapter les modèles pour différentes applications sans avoir besoin de les réentraîner.
- **Rapidité** : Les modèles optimisés avec OpenVINO bénéficient d'une exécution plus rapide, notamment pour des applications de reconnaissance de caractères, de détection d’objets, ou de segmentation.

#### Inconvénients pour la tâche
- **Cible différents** : Le modèle est surtout entraîné pour reconnaître une ligne de caractères chinois avec une largeur de 2000 pixels. Pourtant, nous essayons de reconnaître dans ce travail les caractères un par un au lieu de ligne par ligne. Pour cela, nous avons ajouté du bord blanc à chaque image de caractères pour obtenir une largeur de 2000 pixels, ce qui pourrait biaiser la prédiction du modèle. Par exemple, le modèle n'est pas capable de reconnaître aucune image du caractère "一", qui est probablement considérée comme un tiret selon lui.
- **Absence de contexte linguistique** : Le modèle n'est basé sur aucun modèle du contexte linguistique ce qui pourrait limiter sa performance.
- **Diversité d'écriture** : La reconnaissance des textes manuscrits est difficile à cause de la grande diversité de styles d'écriture.
- **Complexité des caractères chinois** : Les caractères chinois sont très complexes et contiennent de nombreux traits, ce qui rend la tâche d'extraction de caractéristiques particulièrement pénible.

#### Justification
La raison pour laquelle il est intéressant d'employer le modèle **handwritten-simplified-chinese-recognition-0001** provenant de l'OpenVINO ici s'explique par son corpus d'entraînement, qui est des images du manuscrit chinois. Cela permet de former un groupe de comparaison avec notre modèle CRNN, qui est également entraîné sur les images du manuscrit chinois.


## Utilisation

### Génération et Prétraitement du Corpus

Exécutez `pre_processing.ipynb` pour préparer et corriger les images du corpus manuscrit, notamment en affinant le corpus d’images, en segmentant les images en lignes ou caractères individuels, et en nettoyant les images pour optimiser leur qualité visuelle.

### Entraînement du modèle CRNN
Lancez le notebook `CRNN_OCR_Chinese.ipynb` pour entraîner le modèle CRNN sur le jeu de données des caractères manuscrits chinois.

### Comparaison entre Tesseract et EasyOCR
Lancez le notebook `tesseract_vs_easyocr.ipynb` pour évaluer et comparer les performances de Tesseract et EasyOCR sur la tâche d'OCR des caractères chinois.

### Evaluation du modele OpenVINO : handwritten-simplified-chinese-recognition-0001

Lancez le notebook `chinese_handwritten_ocr.ipynb` pour télécharger, configuer le modèle, tester sur les deux corpus de test et évaluer sa performance. 

Avec les résultats de prédiction et les références pré-sauvegardés, il est possible de sauter l'étape de l'utilisation du modèle et charger directement les prédictions et les références pour l'évaluation.

## Résultats
Les résultats de chaque modèle sont sauvegardés sous forme de rapports texte et d'images, incluant les métriques de classification et la matrice de confusion des prédictions.
