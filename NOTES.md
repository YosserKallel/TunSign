# Carnet de Bord - TunSign Partie 1
**Étudiant :** Projet de Machine Learning supervisé
**Enseignante :** Mme Ahlem Arfaoui
**Sujet :** Langue des Signes Tunisienne (TSL / TunSL)

## Étape 1 : Compréhension du problème métier

### Ce qui a été fait
- Le besoin métier est de traduire les signes tunisien en texte écrit.
- La tâche est une classification multiclasse supervisée.
- Le dataset original contient 4423 images réparties en 57 signes textuels.
- Les KPI sont l'Accuracy, le Macro-F1, la matrice de confusion et le temps d'inférence.

### Questions orales
* **Pourquoi l'apprentissage supervisé ?**
  Chaque image possède déjà une étiquette correspondant au signe réel.
* **Pourquoi l'Accuracy seule ne suffit-elle pas ?**
  Les classes sont déséquilibrées. Le Macro-F1 donne une importance égale à chaque signe.
* **Pourquoi faut-il une matrice de confusion ?**
  Elle montre exactement quels signes sont confondus.

## Étape 2 : Collecte des données

### Ce qui a été fait
- Les images viennent du dataset Tunisian Sign Language.
- Les données sont organisées dans 5 catégories : Demandes, Destinations, Famille, Jours et Transport.
- Les 4423 chemins ont été structurés dans un DataFrame pandas avec : chemin_image, categorie et classe.

### Questions orales
* **Quelle est la source des données ?**
  Le First-ever Tunisian Sign Language Dataset, créé par SUP'COM avec l'Association Tunisienne des Sourds.
* **Quel format de données utilise le projet ?**
  Les images brutes sont transformées en coordonnées numériques avec MediaPipe.
* **Combien y a-t-il de signes dans le dataset original ?**
  57 signes. La partie détectable actuelle contient 56 classes parce que le signe « métro » n'a aucun landmark détecté.

## Étape 3 : Exploration et nettoyage

### Ce qui a été fait
- Le DataFrame initial contient 4423 lignes et 3 colonnes.
- Les 4423 chemins sont uniques.
- Les métadonnées ne contiennent aucune valeur manquante.
- L'effectif moyen est de 77.6 images par classe.
- La distribution montre un déséquilibre entre les classes.
- L'IQR doit être expliqué selon les diapositives 42 à 44.
- Le Z-score doit être expliqué selon la diapositive 46.
- Le déséquilibre, SMOTE et class_weight sont expliqués par les diapositives 36 à 39 du cours de préparation des données.

### Questions orales
* **Pourquoi utiliser describe(include='all') ?**
  Cette option permet d'analyser aussi les colonnes textuelles.
* **Pourquoi les classes sont-elles déséquilibrées ?**
  Certaines situations ou certains signes ont été enregistrés plus souvent que d'autres.
* **Comment calculer les bornes IQR ?**
  Q1 - 1.5 × IQR pour la borne basse et Q3 + 1.5 × IQR pour la borne haute.

## Étape 4 : Feature engineering

### Ce qui a été fait
- MediaPipe Hands extrait 21 points 3D pour chaque main.
- Deux mains donnent 2 × 21 × 3 = 126 caractéristiques.
- Les images sans main ont été supprimées.
- Après la nouvelle extraction, 4012 images sont utilisables.
- 411 images ne contiennent aucun landmark détectable.
- 56 classes restent détectables.
- Le signe « métro » fait partie des 57 signes originaux, mais aucune observation n'a été détectée avec MediaPipe.

### Normalisation personnelle
Pour améliorer la démo et rendre les données indépendantes de la position et de la taille :
- les coordonnées du poignet, landmark 0, sont soustraites de tous les points ;
- chaque main est divisée par la plus grande distance poignet-point ;
- les mains sont triées de gauche à droite selon leur position moyenne en x ;
- une seconde main absente garde 63 valeurs égales à zéro.

**Cette normalisation est mon choix personnel. Elle ne vient pas du cours de Mme Arfaoui.**

### Format final
- X : matrice numérique de 4012 observations × 126 caractéristiques.
- y : étiquettes transformées par LabelEncoder.
- 56 classes sont encodées dans la matrice actuelle.

### Questions orales
* **Pourquoi MediaPipe au lieu des pixels bruts ?**
  Il réduit 150 528 pixels à 126 coordonnées utiles et évite le Deep Learning interdit par le cours.
* **Pourquoi supprimer les images sans main ?**
  Elles ne contiennent aucun landmark exploitable. Imputer des gestes fictifs serait moins fiable.
* **Pourquoi normaliser par rapport au poignet ?**
  C'est un choix personnel pour réduire l'effet de la position de la main.
