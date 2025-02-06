import os
from PIL import Image
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import imagehash
from PIL import ImageStat
import shutil

class PhotoAnalyzer:
    def __init__(self, input_folder, output_folder):
        self.input_folder = input_folder
        self.output_folder = output_folder
        # Carica il modello ResNet50 pre-addestrato
        self.model = ResNet50(weights='imagenet', include_top=False)
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def analyze_photo(self, image_path):
        """Analizza una singola foto e restituisce un punteggio di qualità."""
        try:
            # Apri l'immagine
            img = Image.open(image_path)
            
            # Controllo 1: Risoluzione
            width, height = img.size
            resolution_score = min(width * height / (4000 * 3000), 1.0)  # Normalizza per una risoluzione target
            
            # Controllo 2: Nitidezza
            gray_img = img.convert('L')
            stat = ImageStat.Stat(gray_img)
            sharpness_score = stat.var[0] / 10000  # Normalizza la varianza
            
            # Controllo 3: Esposizione
            exposure_score = self._check_exposure(img)
            
            # Controllo 4: Analisi estetica con ResNet
            aesthetic_score = self._get_aesthetic_score(img)
            
            # Calcola il punteggio finale
            final_score = (resolution_score * 0.2 + 
                         sharpness_score * 0.3 + 
                         exposure_score * 0.2 + 
                         aesthetic_score * 0.3)
            
            return final_score
            
        except Exception as e:
            print(f"Errore nell'analisi di {image_path}: {str(e)}")
            return 0

    def _check_exposure(self, img):
        """Valuta l'esposizione dell'immagine."""
        stat = ImageStat.Stat(img)
        mean_brightness = sum(stat.mean) / len(stat.mean)
        # Punteggio più alto per valori medi di luminosità
        return 1.0 - abs(mean_brightness - 128) / 128

    def _get_aesthetic_score(self, img):
        """Analizza l'estetica dell'immagine usando ResNet."""
        # Ridimensiona l'immagine per il modello
        img_resized = img.resize((224, 224))
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Ottieni le feature
        features = self.model.predict(x)
        # Calcola un punteggio basato sulla complessità delle feature
        score = np.mean(features) + np.std(features)
        # Normalizza il punteggio
        return (score - -0.5) / (0.5 - -0.5)  # Normalizza tra 0 e 1

    def process_folder(self, threshold=0.7):
        """Processa tutte le foto nella cartella di input."""
        photo_scores = []
        
        # Analizza tutte le foto
        for root, _, files in os.walk(self.input_folder):
            for img_name in files:
                if img_name.lower().endswith('.jpg'):
                    img_path = os.path.join(root, img_name)
                    image_path = os.path.join(self.input_folder, img_path)
                    score = self.analyze_photo(image_path)
                    photo_scores.append((img_path, score))
        
        # Ordina le foto per punteggio
        photo_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Copia le migliori foto nella cartella di output
        for filename, score in photo_scores:
            if score >= threshold:
                src = os.path.join(self.input_folder, filename)
                relative_path = os.path.relpath(src, self.input_folder)
                dst = os.path.join(self.output_folder, relative_path.replace(os.sep, '_').replace(' ', '-'))

                shutil.copy2(src, dst)
                print(f"Copiata {filename} con punteggio {score:.2f}")

# Esempio di utilizzo
if __name__ == "__main__":
    input_folder = "/Users/erighetto/Library/CloudStorage/Dropbox/Photos/2025"
    output_folder = "/Users/erighetto/Pictures/Selections"
    os.makedirs(output_folder, exist_ok=True)
    
    analyzer = PhotoAnalyzer(input_folder, output_folder)
    analyzer.process_folder(threshold=1.3)  # Modifica la soglia in base alle tue esigenze