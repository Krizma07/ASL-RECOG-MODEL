# ASL Recognition Model - projekt na potrzeby pracy inżynierskiej

Projekt realizuje rozpoznawanie języka migowego (American Sign Language - ASL) przy użyciu sieci konwolucyjnej w TensorFlow.

## Spis treści
1. [Opis projektu](#opis-projektu)
2. [Zestaw danych](#zestaw-danych)
3. [Wymagania](#wymagania)
4. [Instrukcja instalacji](#instrukcja-instalacji)
5. [Sposób użycia](#sposób-użycia)
6. [Autor](#autor)

## Opis projektu
Ten projekt klasyfikuje obrazy przedstawiające litery ASL na podstawie zestawu danych. Model został zbudowany przy użyciu frameworka TensorFlow i zapisany w formacie `.keras`. 

## Zestaw danych
Do tego projektu wykorzystano zestaw danych pochodzący z Kaggle:  
[Sign Language MNIST Dataset](https://www.kaggle.com/datasets/datamunge/sign-language-mnist?select=sign_mnist_test)

Zestaw danych zawiera obrazy przedstawiające litery alfabetu migowego w formacie grayscale 28x28 pikseli. Obrazy zostały wyeksportowane z plików CSV, które znajdowały się w podanym źródle. W projekcie zestawy danych podzielono na:

- **TRAIN**: 27 455 obrazów  
- **TEST**: 7 172 obrazy

Wszystkie obrazy zostały odpowiednio przetworzone i skategoryzowane do 24 klas, reprezentujących litery alfabetu migowego (bez znaków 'J' i 'Z', które wymagają ruchu).

## Wymagania
Do uruchomienia projektu wymagane są:
- Python 3.10
- TensorFlow
- Matplotlib
- NumPy
- SciPy

## Instrukcja instalacji
1. Sklonuj repozytorium:
   ```bash
   git clone still in progress
   cd asl_recognition

2. Instalacja zależności

Utwórz i aktywuj środowisko wirtualne, a następnie zainstaluj wymagane pakiety:

python -m venv venv
source venv/bin/activate    # Na systemach Linux/Mac
venv\Scripts\activate       # Na Windows
pip install -r requirements.txt

## Sposób użycia
1. Trenowanie modelu

Aby rozpocząć trenowanie modelu, uruchom skrypt:

python train_model.py
Model zostanie zapisany w folderze models jako plik asl_model.keras.

2. Wizualizacja wyników
Po zakończeniu trenowania wygenerowane zostaną dwa wykresy (dokładność i strata). Te wykresy automatycznie zapisują się w folderze models jako pliki accuracy_plot.png oraz loss_plot.png.

3. Załadowanie wytrenowanego modelu
Aby załadować wytrenowany model w innym skrypcie Python, użyj:

from tensorflow.keras.models import load_model
model = load_model('models/asl_model.keras')

## Autor
- **Krzysztof Stolc**
