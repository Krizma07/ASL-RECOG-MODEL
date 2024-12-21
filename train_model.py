import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import json

# Ścieżki do datasetów (treningowy i testowy)
train_dir = 'dataset/train'
test_dir = 'dataset/test'

# Inicjalizacja generatorów danych
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,  # Normalizacja pikseli do zakresu [0, 1]
    rotation_range=10,  # Losowe obracanie obrazów w zakresie 10 stopni
    width_shift_range=0.1,  # Losowe przesunięcie obrazu w poziomie
    height_shift_range=0.1,  # Losowe przesunięcie obrazu w pionie
    shear_range=0.1,  # Losowe zniekształcanie obrazu
    zoom_range=0.1,  # Losowe powiększanie obrazu
    horizontal_flip=False  # Brak lustrzanego odbicia
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)  # Tylko normalizacja dla zbioru testowego

# Generator danych dla zbioru treningowego
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28, 28),  # Zmiana rozmiaru obrazu na 28x28
    color_mode='grayscale',  # Wczytywanie obrazów w skali szarości
    batch_size=32,  # Rozmiar partii (batch)
    class_mode='categorical',  # Modele wieloklasowe (24 klasy)
    shuffle=True  # Losowe mieszanie danych
)

# Generator danych dla zbioru testowego
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(28, 28),  # Zmiana rozmiaru obrazu na 28x28
    color_mode='grayscale',  # Wczytywanie obrazów w skali szarości
    batch_size=32,  # Rozmiar partii (batch)
    class_mode='categorical',  # Modele wieloklasowe (24 klasy)
    shuffle=False  # Brak losowego mieszania dla zbioru testowego (ważne do oceny)
)

# Model Sequential z dodatkowymi warstwami i regularyzacją L2
model = Sequential([
    Input(shape=(28, 28, 1)),  # Wejście obrazu o rozmiarze 28x28 i 1 kanale (czarno-białe)
    Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)),  # Pierwsza warstwa konwolucyjna
    MaxPooling2D((2, 2)),  # Warstwa max-pooling, redukuje wymiary obrazu
    Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.0005)),  # Druga warstwa konwolucyjna
    MaxPooling2D((2, 2)),  # Druga warstwa max-pooling
    Flatten(),  # Spłaszczenie wyników z warstw konwolucyjnych do wektora
    Dense(128, activation='relu', kernel_regularizer=l2(0.0005)),  # Warstwa w pełni połączona
    Dropout(0.5),  # Warstwa Dropout, aby uniknąć przeuczenia (50% szansa na wyłączenie neuronu)
    Dense(24, activation='softmax')  # Ostatnia warstwa do klasyfikacji na 24 klasy
])

# Kompilacja modelu z użyciem optymalizatora Adam i funkcji strat 'categorical_crossentropy'
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Optymalizator Adam z niskim współczynnikiem uczenia
    loss='categorical_crossentropy',  # Funkcja strat dla wieloklasowej klasyfikacji
    metrics=['accuracy']  # Mierzymy dokładność modelu
)

# Wczesne zatrzymanie (Early Stopping) - zatrzymanie treningu, jeśli walidacja nie poprawia się przez 5 epok
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Trenowanie modelu
history = model.fit(
    train_generator,  # Dane treningowe
    steps_per_epoch=len(train_generator),  # Liczba kroków na epokę (ilość batchy w jednym przejściu)
    epochs=30,  # Liczba epok
    validation_data=test_generator,  # Dane walidacyjne
    validation_steps=len(test_generator),  # Liczba kroków na epokę dla danych walidacyjnych
    callbacks=[early_stopping]  # Użycie Early Stopping
)

# Zapisywanie modelu do pliku
model.save('models/asl_model.keras')

# Zapisanie historii treningu (dokładność, strata) do pliku JSON
with open('models/history.json', 'w') as f:
    json.dump(history.history, f)

# Wizualizacja wyników - dokładność (Accuracy)
plt.plot(history.history['accuracy'], label='Train Accuracy')  # Wykres dokładności dla treningu
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')  # Wykres dokładności dla walidacji
plt.xlabel('Epochs')  # Etykieta osi X
plt.ylabel('Accuracy')  # Etykieta osi Y
plt.legend(loc='lower right')  # Pozycja legendy
plt.title('Model Accuracy')  # Tytuł wykresu
plt.savefig('models/accuracy_plot.png')  # Zapisuje wykres dokładności
plt.show()  # Wyświetlenie wykresu
plt.close()  # Zamknięcie wykresu

# Wizualizacja wyników - strata (Loss)
plt.plot(history.history['loss'], label='Train Loss')  # Wykres straty dla treningu
plt.plot(history.history['val_loss'], label='Validation Loss')  # Wykres straty dla walidacji
plt.xlabel('Epochs')  # Etykieta osi X
plt.ylabel('Loss')  # Etykieta osi Y
plt.legend(loc='upper right')  # Pozycja legendy
plt.title('Model Loss')  # Tytuł wykresu
plt.savefig('models/loss_plot.png')  # Zapisuje wykres straty
plt.show()  # Wyświetlenie wykresu
plt.close()  # Zamknięcie wykresu
