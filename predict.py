import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Załaduj model
model = load_model('models/asl_model.keras')

# Mapowanie klas do liter (pomijając J i Z)
class_mapping = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 
    8: 'I', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 
    23: 'X', 24: 'Y'
}

# Funkcja do przewidywania na podstawie obrazu
def predict_image(image_path):
    # Załaduj obraz (kolorowy JPG, o rozmiarze 200x200)
    img = Image.open(image_path)
    
    # Zmiana rozmiaru obrazu na 28x28px
    img = img.resize((28, 28))
    
    # Konwersja na odcienie szarości
    img = img.convert('L')
    
    # Przekształcenie obrazu na tablicę numpy i normalizacja
    img_array = img_to_array(img) / 255.0  # Normalizacja do zakresu 0-1
    img_array = np.expand_dims(img_array, axis=0)  # Dodanie wymiaru batch
    
    # Dokonaj predykcji
    prediction = model.predict(img_array)
    
    # Uzyskaj indeks klasy
    predicted_class = np.argmax(prediction, axis=1)[0]
    
    # Mapowanie na literę
    predicted_letter = class_mapping.get(predicted_class, "Unknown")
    
    return predicted_letter, prediction[0][predicted_class]

# Testowanie na pojedynczym obrazie
image_path = 'randomimages\R6.jpg'  # Podaj pełną ścieżkę do obrazu
predicted_letter, confidence = predict_image(image_path)

print(f"Predykcja: {predicted_letter}, Pewność: {confidence:.2f}")
