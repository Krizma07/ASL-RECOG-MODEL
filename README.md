# ASL Sign Language Recognition Model

## Overview
This project focuses on building a deep learning model to recognize American Sign Language (ASL) alphabet using static images. The model is trained using the ASL MNIST dataset, which consists of grayscale images of size 28x28 pixels representing the letters of the ASL alphabet (excluding letters J and Z).

## Dataset
The dataset used in this project is ASL MNIST, which contains grayscale images of size 28x28 pixels. The dataset is divided into training and testing directories:

dataset/train/: Training images
dataset/test/: Testing images
Each image corresponds to a specific letter in the ASL alphabet, and the model is tasked with classifying these images into one of 24 classes (A-Z, excluding J and Z).

The dataset is sourced from this Kaggle repository. The images were exported from .csv files available at the above link and have been organized into corresponding folders for each letter. For example, images representing the sign for the letter "A" are placed in the A folder, and so on for each letter (excluding J and Z).

## Requirements
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- JSON

You can install the necessary dependencies by running:

pip install -r requirements.txt

## Model Architecture
The model uses a **Convolutional Neural Network (CNN)** to perform the image classification task. The architecture consists of:
1. **Convolutional Layers:** Two layers of 32 and 64 filters, each with a kernel size of (3, 3), followed by **MaxPooling**.
2. **Dropout Layer:** A dropout layer with a rate of 0.5 is used to prevent overfitting.
3. **Fully Connected Layer:** A dense layer with 128 neurons.
4. **Output Layer:** A dense layer with 24 units, using **Softmax activation** for multi-class classification.

Regularization (L2) is applied to the convolutional and dense layers to avoid overfitting.

## Training
The model is trained with the following parameters:
- Optimizer: **Adam**
- Learning Rate: **0.0001**
- Loss Function: **Categorical Crossentropy**
- Metrics: **Accuracy**
- Epochs: 30
- Batch Size: 32
- Early Stopping: The training process will stop early if the validation loss does not improve after 5 epochs.

## Results
The model achieved the following performance on the test dataset:

- **Training Accuracy:** ~87.5% 
- **Validation Accuracy:** ~97.4%

The model shows consistent improvement across epochs, starting with low accuracy and gradually improving as the training progresses.

## Model Saving
The trained model is saved as `asl_model.keras` in the `models/` directory.

# Load the trained model
model = load_model('models/asl_model.keras')

## Visualization
The training and validation accuracy and loss are visualized using Matplotlib. The accuracy and loss plots are saved as `accuracy_plot.png` and `loss_plot.png`, respectively.

## Improvements
While the model achieves a good level of accuracy, the following improvements can be explored:
- **Increasing Epochs:** Training for more epochs could potentially improve the model’s accuracy further.
- **Hyperparameter Tuning:** Experiment with different values of learning rate, L2 regularization, and batch size.
- **Augmenting Data:** Add more data augmentation techniques such as rotation, brightness adjustment, and noise addition.

## Conclusion
This model serves as a robust baseline for ASL sign language recognition tasks. With further experimentation and optimization, it can be improved to achieve even higher accuracy.

# Model Rozpoznawania Języka Migowego ASL

## Przegląd
Celem tego projektu jest stworzenie modelu głębokiego uczenia do rozpoznawania alfabetu Amerykańskiego Języka Migowego (ASL) przy użyciu statycznych obrazów. Model jest trenowany na bazie datasetu ASL MNIST, który składa się z szarych obrazów o rozmiarze 28x28 pikseli przedstawiających litery alfabetu ASL (z wyłączeniem liter J i Z).

## Zbiór Danych
Zestaw danych użyty w tym projekcie to ASL MNIST, który zawiera obrazy w odcieniach szarości o rozmiarze 28x28 pikseli. Zestaw danych jest podzielony na katalogi treningowe i testowe:

dataset/train/: Obrazy treningowe
dataset/test/: Obrazy testowe
Każdy obraz odpowiada konkretnej literze w alfabecie ASL, a model ma za zadanie sklasyfikować te obrazy do jednej z 24 klas (A-Z, z wyłączeniem liter J i Z).

Zestaw danych pochodzi z tego repozytorium na Kaggle. Obrazy zostały wyeksportowane z plików .csv dostępnych pod tym linkiem, a następnie posegregowane do odpowiednich folderów odpowiadających poszczególnym literom. Na przykład, obrazy przedstawiające znak dla litery "A" znajdują się w folderze A, i tak dalej dla każdej litery (z wyłączeniem J i Z).

## Wymagania
- Python 3.x
- TensorFlow 2.x
- Keras
- Matplotlib
- JSON

Aby zainstalować wymagane zależności, użyj polecenia:

pip install -r requirements.txt

## Architektura Modelu
Model wykorzystuje **Sieć Neuronową Splotową (CNN)** do wykonywania zadania klasyfikacji obrazów. Architektura składa się z:
1. **Warstwy Splotowej:** Dwie warstwy z 32 i 64 filtrami o rozmiarze jądra (3, 3), każda z następującą po niej **Warstwą MaxPooling**.
2. **Warstwa Dropout:** Warstwa dropout o współczynniku 0.5, stosowana w celu zapobiegania nadmiernemu dopasowaniu.
3. **Warstwa W pełni Połączona:** Warstwa gęsta z 128 neuronami.
4. **Warstwa Wyjściowa:** Warstwa gęsta z 24 jednostkami, używająca aktywacji **Softmax** do klasyfikacji wieloklasowej.

Zastosowano regularyzację (L2) w warstwach splotowych oraz gęstych, aby uniknąć nadmiernego dopasowania.

## Trenowanie
Model jest trenowany z następującymi parametrami:
- Optymalizator: **Adam**
- Współczynnik uczenia: **0.0001**
- Funkcja straty: **Categorical Crossentropy**
- Metryki: **Accuracy**
- Epoki: 30
- Rozmiar partii: 32
- Wczesne zatrzymanie: Proces trenowania zatrzyma się wcześniej, jeśli strata walidacyjna nie poprawi się przez 5 epok.

## Wyniki
Model osiągnął następujące wyniki na zbiorze testowym:

- **Dokładność treningowa:** ~87.5%
- **Dokładność walidacyjna:** ~97.4%

Model wykazuje stałą poprawę w trakcie treningu, począwszy od niskiej dokładności, a następnie stopniowo poprawiając się w miarę postępu treningu.

## Zapisanie Modelu
Po treningu model jest zapisywany jako `asl_model.keras` w katalogu `models/`.

# Załaduj wytrenowany model
model = load_model('models/asl_model.keras')

## Wizualizacja
Dokładność oraz strata treningowa i walidacyjna są wizualizowane przy użyciu Matplotlib. Wykresy dokładności i straty są zapisywane jako `accuracy_plot.png` oraz `loss_plot.png`.

## Możliwości Udoskonalenia
Mimo że model osiąga dobrą dokładność, istnieje kilka możliwości jego poprawy:
- **Zwiększenie liczby epok:** Trening przez więcej epok może potencjalnie poprawić dokładność modelu.
- **Dopasowanie hiperparametrów:** Eksperymentowanie z różnymi wartościami współczynnika uczenia, regularyzacji L2 oraz rozmiaru partii.
- **Augmentacja Danych:** Dodanie nowych technik augmentacji danych, takich jak obrót, zmiana jasności i dodawanie szumu.

## Podsumowanie
Ten model stanowi solidną bazę do rozpoznawania języka migowego ASL. Poprzez dalsze eksperymenty i optymalizację, można go ulepszyć, aby uzyskać jeszcze wyższą dokładność.