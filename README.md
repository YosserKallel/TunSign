# TunSign — Partie 1

Projet académique de reconnaissance de la Langue des Signes Tunisienne (TSL / TunSL).

Cette livraison présente uniquement les **Étapes 1 à 4** :

1. Compréhension du problème métier.
2. Collecte des données.
3. Exploration et nettoyage EDA.
4. Feature engineering avec MediaPipe.

Les Étapes 5 à 7 seront réalisées plus tard par le binôme.

## Règle pédagogique

Le projet doit suivre le cours de Mme Ahlem Arfaoui :
- pandas
- numpy
- matplotlib
- scikit-learn
- pas de Deep Learning

## Données

Le dataset source est le *First-ever Tunisian Sign Language Dataset*.

- 4423 images originales.
- 57 signes dans le dataset original.
- 4012 images exploitables après détection MediaPipe.
- 411 images rejetées sans main détectée.
- 56 classes détectables.
- La classe « métro » fait partie du dataset original, mais aucun landmark n'a été détecté.

## Extraction des caractéristiques

Chaque observation contient 126 caractéristiques :
- 63 valeurs pour la première main : h1_x0 à h1_z20.
- 63 valeurs pour la seconde main : h2_x0 à h2_z20.

Chaque main est :
- centrée sur son poignet, landmark 0 ;
- divisée par sa taille maximale ;
- placée dans l'ordre gauche à droite.

Une seconde main absente est représentée par 63 zéros.

Cette normalisation est un choix personnel du projet, pas une règle du cours.

## Fichiers principaux

- notebooks/tunsign.ipynb : étapes 1 à 4.
- extract_landmarks.py : génération des landmarks normalisés.
- data/landmarks.csv : matrice des 4012 observations.
- data/processed/ : copie de travail des données traitées.
- models/hand_landmarker.task : modèle MediaPipe.
- NOTES.md : explications et préparation orale.
- HANDOFF_BINOME.md : travail restant pour le binôme.
- course/ : cours et PDFs fournis par Mme Arfaoui.

## Source du dataset

Première version publique du Tunisian Sign Language Dataset, créée par Wassim Chakroun et Rihab Jerbi de SUP'COM avec l'Association Tunisienne des Sourds.

## Références utilisées comme inspiration

Les fichiers déposés dans les références ont été consultés uniquement comme inspiration :
- real-time-sign-language-translator
- SIGNLAN-TO-TEXT
- Sign_recognition_system
- real-time-sign-language-recognition
- TunSL_detection

Aucun code de Deep Learning n'a été ajouté.
