import time
import os
import pandas as pd
import numpy as np
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Chemins
base_dir = Path(r'c:\Users\Yosser\Desktop\TunSign')
raw_dir = base_dir / 'data' / 'raw'
data_dir = list(raw_dir.rglob('Data'))[0]
model_path = base_dir / 'models' / 'hand_landmarker.task'

csv_dir = base_dir / 'data'
csv_dir.mkdir(parents=True, exist_ok=True)
csv_path = csv_dir / 'landmarks.csv'

# Configuration MediaPipe Hands (Tasks API)
options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=str(model_path)),
    num_hands=2,
    min_hand_detection_confidence=0.1
)
detector = vision.HandLandmarker.create_from_options(options)

# Noms des 126 colonnes de caractéristiques
colonnes_features = []
for h in [1, 2]:
    for i in range(21):
        for coord in ['x', 'y', 'z']:
            colonnes_features.append(f'h{h}_{coord}{i}')

print(f"Total colonnes de caractéristiques : {len(colonnes_features)}")

images = sorted(list(data_dir.rglob('*.jpg')))
total_imgs = len(images)
print(f"Extraction des landmarks pour {total_imgs} images...")

records = []
non_detectes = 0
start_time = time.time()

for idx, img_path in enumerate(images):
    if (idx + 1) % 500 == 0 or idx == total_imgs - 1:
        elapsed = time.time() - start_time
        fps = (idx + 1) / elapsed
        print(f"[{idx + 1}/{total_imgs}] {fps:.1f} img/s - non détectées : {non_detectes}")

    try:
        mp_image = mp.Image.create_from_file(str(img_path))
        result = detector.detect(mp_image)
    except Exception as e:
        non_detectes += 1
        continue

    # Si aucune main n'est détectée -> valeur manquante (on ignore la ligne)
    if not result.hand_landmarks or len(result.hand_landmarks) == 0:
        non_detectes += 1
        continue

    # Vecteur de 126 caractéristiques initialisé à zéro
    features = [0.0] * 126

    # Tri stable des mains de gauche à droite selon leur position horizontale
    mains_triees = sorted(result.hand_landmarks, key=lambda main: sum(pt.x for pt in main) / len(main))

    # Normalisation indépendante de chaque main présente
    for main_index, main in enumerate(mains_triees[:2]):
        offset = main_index * 63
        wrist = main[0]
        wrist_x = float(wrist.x)
        wrist_y = float(wrist.y)
        wrist_z = float(wrist.z)

        relative_points = []
        for pt in main:
            relative_x = float(pt.x) - wrist_x
            relative_y = float(pt.y) - wrist_y
            relative_z = float(pt.z) - wrist_z
            relative_points.append((relative_x, relative_y, relative_z))

        hand_size = max(
            np.sqrt(relative_x * relative_x + relative_y * relative_y + relative_z * relative_z)
            for relative_x, relative_y, relative_z in relative_points
        )

        if hand_size == 0:
            hand_size = 1.0

        for i, (relative_x, relative_y, relative_z) in enumerate(relative_points):
            features[offset + i * 3] = relative_x / hand_size
            features[offset + i * 3 + 1] = relative_y / hand_size
            features[offset + i * 3 + 2] = relative_z / hand_size

    record = {
        'chemin_image': str(img_path),
        'categorie': img_path.parent.parent.name,
        'classe': img_path.parent.name
    }
    for col_name, val in zip(colonnes_features, features):
        record[col_name] = val

    records.append(record)

df_landmarks = pd.DataFrame(records)
df_landmarks.to_csv(csv_path, index=False)

# Sauvegarde également dans data/processed pour flexibilité
processed_dir = base_dir / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)
df_landmarks.to_csv(processed_dir / 'landmarks.csv', index=False)

total_time = time.time() - start_time
print(f"\nExtraction terminée en {total_time:.1f} secondes !")
print(f"Images traitées avec succès : {len(df_landmarks)}")
print(f"Images rejetées (aucune main détectée) : {non_detectes}")
print(f"Fichier sauvegardé dans : {csv_path}")
