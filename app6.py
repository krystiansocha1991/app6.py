import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import ModelCheckpoint
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import tensorflow as tf

# Ścieżka do folderu z artykułami
articles_folder = "/home/krystian/nowymodel/hacki"
model_path = '/home/krystian/nowymodel/modelhacki/model.h5'

# Inicjalizacja tokenizer'a
tokenizer = keras.preprocessing.text.Tokenizer()

# Wczytanie zawartości artykułów
articles = []
for filename in os.listdir(articles_folder):
    filepath = os.path.join(articles_folder, filename)
    with open(filepath, "r", encoding="utf-8") as file:
        content = file.read()
        articles.append(content)

# Tokenizacja tekstu
tokenizer.fit_on_texts(articles)
vocab_size = len(tokenizer.word_index) + 1
print("vocab_size:", vocab_size)

# Usunięcie stop words
stop_words = set(stopwords.words("english"))
filtered_articles = []
for article in articles:
    tokens = word_tokenize(article)
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    filtered_article = " ".join(filtered_tokens)
    filtered_articles.append(filtered_article)

# Podział tekstu na sekwencje
input_sequences = tokenizer.texts_to_sequences(filtered_articles)
output_sequences = [seq[1:] for seq in input_sequences]
input_sequences = [seq[:-1] for seq in input_sequences]

# Ograniczenie długości sekwencji
max_length = 5000  # Długość docelowa sekwencji
input_sequences = pad_sequences(input_sequences, maxlen=max_length, truncating='post')
output_sequences = pad_sequences(output_sequences, maxlen=max_length, truncating='post')

# Normalizacja danych wejściowych
scaler = MinMaxScaler()
input_sequences_normalized = scaler.fit_transform(input_sequences)

# Przygotowanie danych treningowych
X = np.array(input_sequences)
y = np.array(output_sequences)

# Sprawdzenie rozmiarów sekwencji
if X.shape[0] == y.shape[0]:
    print("Rozmiary sekwencji wejściowych i wyjściowych po skalowaniu są takie same.")
else:
    print("Rozmiary sekwencji wejściowych i wyjściowych po skalowaniu są różne.")

logits_shape = X.shape
labels_shape = y.shape

print("Kształt logitów:", logits_shape)
print("Kształt etykiet:", labels_shape)

# Definicja modelu
input_layer = Input(shape=(max_length,))
embedding_layer = Embedding(vocab_size, 100)(input_layer)
lstm_layer = LSTM(1)(embedding_layer)
output_layer = Dense(5000, activation="softmax")(lstm_layer)
model = Model(input_layer, output_layer)

# Kompilacja modelu
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Callback do zapisu modelu
checkpoint = ModelCheckpoint(model_path, monitor="loss", savebest_only=True)

initial_batch_size = 1
batch_size_increase_epoch = 10
batch_size_increase_value = 1
num_epochs = 100
batch_size = initial_batch_size

for epoch in range(num_epochs):
    if epoch > 0 and epoch % batch_size_increase_epoch == 0:
        batch_size += batch_size_increase_value
    
    history = model.fit(X, y, epochs=1, batch_size=batch_size, callbacks=[checkpoint])
    accuracy = history.history['accuracy'][0]
    print(f"Epoch {epoch+1} - Loss: {history.history['loss'][0]} - Accuracy: {accuracy}")
    
    # Funkcja zwrotna do zmiany learning rate co 5 epok
    def update_learning_rate(epoch, lr):
        if epoch % 5 == 0 and epoch > 0:
            new_learning_rate = lr * 0.1  # Modyfikuj learning rate (np. zmniejsz o 10%)
            optimizer.learning_rate.assign(new_learning_rate)
            print(f'Learning rate changed to: {new_learning_rate}')
    
    # Zapisywanie modelu co 10 epok
    if (epoch + 1) % 10 == 0:
        model.save(model_path)
        print("Model zapisany.")

# Zapisanie wytrenowanego modelu
model.save("/home/krystian/nowymodel/modelhacki/model.h5")
print("Model zapisany.")

