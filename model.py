import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the dataset
with open("dataset.txt", "r") as file:
    dataset = [line.strip().split('\t') for line in file.readlines()]

# Apply Caesar cipher to the dataset
def caesar_cipher(text, shift):
    result = ""
    for char in text:
        if char.isalpha():
            shifted = ord(char) + shift
            if char.islower():
                if shifted > ord('z'):
                    shifted -= 26
                elif shifted < ord('a'):
                    shifted += 26
            elif char.isupper():
                if shifted > ord('Z'):
                    shifted -= 26
                elif shifted < ord('A'):
                    shifted += 26
            result += chr(shifted)
        else:
            result += char
    return result

dataset = [(input_text, caesar_cipher(target_text, 5)) for input_text, target_text in dataset]

# Splitting the dataset into input and target
input_texts = [pair[0] for pair in dataset]
target_texts = [pair[1] for pair in dataset]

# Creating character dictionaries
input_characters = sorted(list(set(''.join(input_texts))))
target_characters = sorted(list(set(''.join(target_texts))))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)
max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

# Load token indices and max sequence lengths from the file
token_data = np.load("token_indices.npz", allow_pickle=True)
input_token_index = dict(token_data["input_token_index"].tolist())
target_token_index = dict(token_data["target_token_index"].tolist())

# Creating encoder and decoder data
encoder_input_data = np.zeros((len(input_texts), max_encoder_seq_length), dtype="float32")
decoder_input_data = np.zeros((len(input_texts), max_decoder_seq_length), dtype="float32")
decoder_target_data = np.zeros((len(input_texts), max_decoder_seq_length, num_decoder_tokens), dtype="float32")

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t] = input_token_index[char]
    for t, char in enumerate(target_text):
        decoder_input_data[i, t] = target_token_index[char]
        if t > 0:
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.0

# Define the model
latent_dim = 256  # Dimensionality of the encoding space

model = Sequential()
model.add(Embedding(num_encoder_tokens, latent_dim))
model.add(LSTM(latent_dim, return_sequences=True))
model.add(Dense(num_decoder_tokens, activation="softmax"))

# Compile the model
model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(encoder_input_data, decoder_target_data, batch_size=64, epochs=30, validation_split=0.2)

# Save the model
model.save("codebreaking_model.keras")

print("Model trained and saved successfully.")
