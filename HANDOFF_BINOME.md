# TunSign — Handoff Partie 2

## Projet

TunSign traduit la Langue des Signes Tunisienne en texte.

 exigence :
- suivre uniquement le cours fourni ;
- utiliser pandas, numpy, matplotlib et scikit-learn ;
- aucune méthode de Deep Learning ;
- respecter les 7 étapes du projet.

## Ce qui est fait (Partie 1, par yosser)

### Étape 1
- Objectif métier et problème de classification multiclasse défini.
- KPI définis : Accuracy, Macro-F1, matrice de confusion et temps d'inférence.

### Étape 2
- Les 4423 images originales ont été recensées.
- Le dataset contient 57 signes dans les données sources.

### Étape 3
- EDA sur les métadonnées.
- Vérification des valeurs manquantes et doublons.
- Analyse de la distribution et déséquilibre des classes.
- IQR et Z-score étudiés.

### Étape 4
- 4012 images ont été exploitées avec MediaPipe.
- 411 images ont été rejetées sans main détectée.
- 56 classes restent détectables.
- Le signe « métro » fait partie des 57 signes, mais aucune image n'a produit de landmarks.
- Fichiers produits :
  - extract_landmarks.py
  - data/landmarks.csv
  - data/processed/landmarks.csv
  - notebooks/tunsign.ipynb

### Format des caractéristiques

Chaque exemple contient 126 valeurs :
- h1_x0 à h1_z20 : main gauche ou première main.
- h2_x0 à h2_z20 : seconde main.

Pour chaque main :
- soustraire le poignet, landmark 0 ;
- diviser tous les points par la taille maximale de la main ;
- trier les mains de gauche à droite.

Si une seconde main manque, ses 63 valeurs restent égales à zéro.

Cette normalisation est un choix personnel pour cette partie. Elle ne vient pas du cours.

## Ce qui reste à faire (Partie 2, par nour islem)

### Étape 5 : Modélisation
- Faire un split 80/20 stratifié.
- Utiliser random_state=42.
- Ajuster StandardScaler uniquement sur X_train.
- Transformer X_test avec les paramètres appris sur X_train.
- Entraîner 5 modèles :
  - LogisticRegression
  - DecisionTree
  - RandomForest
  - SVM
  - KNN
- Utiliser class_weight="balanced" lorsque le modèle le permet.
- Utiliser cross_val_score pour les scores Accuracy et Macro-F1.

### Étape 6 : Évaluation
- Évaluer les cinq modèles sur le jeu Test.
- Calculer Accuracy, Precision Macro, Recall Macro et F1 Macro.
- Produire classification_report et matrice de confusion.
- Comparer les performances Train/Test pour détecter le surapprentissage.

### Étape 7 : Déploiement
- Choisir le meilleur modèle.
- Sauvegarder avec joblib :
  - tunsign_model.joblib
  - standard_scaler.joblib
  - label_encoder.joblib
- Ajouter une démo webcam.
- Les données webcam doivent utiliser exactement le même traitement que extract_landmarks.py.
- Utiliser OpenCV pour l'affichage.
- Garder le même ordre des mains et les mêmes 126 valeurs.

## Répartition du travail

| Tâche | Responsable |
|---|---|
| Étapes 1 à 4 | yosser |
| Étape 5 Modélisation | nour islem |
| Étape 6 Évaluation | nour islem |
| Étape 7 Déploiement et démo webcam | nour islem |

## Exécution

Dans cet ordre :

1. Installer les bibliothèques :
   pip install -r requirements.txt
2. Vérifier les fichiers :
   data/landmarks.csv
   models/hand_landmarker.task
3. Ouvrir notebooks/tunsign.ipynb.
4. Exécuter toutes les cellules Partie 1 de haut en bas.
5. Après cette partie, reprendre depuis l'étape 5.
