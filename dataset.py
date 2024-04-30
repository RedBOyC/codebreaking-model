import random
import string

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

def generate_dataset(num_samples):
    dataset = []
    for _ in range(num_samples):
        text = ''.join(random.choices(string.ascii_letters + string.digits + string.punctuation, k=random.randint(10, 50)))
        encoded_text = caesar_cipher(text, 5)
        dataset.append((text, encoded_text))
    return dataset

def decode_caesar_cipher(encoded_text, shift):
    result = ""
    for char in encoded_text:
        if char.isalpha():
            shifted = ord(char) - shift
            if char.islower():
                if shifted < ord('a'):
                    shifted += 26
            elif char.isupper():
                if shifted < ord('A'):
                    shifted += 26
            result += chr(shifted)
        else:
            result += char
    return result

if __name__ == "__main__":
    # Generate dataset
    dataset = generate_dataset(10000)

    # Save dataset to a file
    with open("dataset.txt", "w") as f:
        for pair in dataset:
            f.write(pair[0] + '\t' + pair[1] + '\n')

    # Decode the dataset
    decoded_dataset = []
    with open("dataset.txt", "r") as f:
        for line in f:
            original_text, encoded_text = line.strip().split('\t')
            decoded_text = decode_caesar_cipher(encoded_text, 5)
            decoded_dataset.append((original_text, decoded_text))

    # Output the decoded dataset
    for pair in decoded_dataset[:10]:  # Displaying the first 10 samples
        print("Original Text:", pair[0])
        print("Decoded Text:", pair[1])
        print()
