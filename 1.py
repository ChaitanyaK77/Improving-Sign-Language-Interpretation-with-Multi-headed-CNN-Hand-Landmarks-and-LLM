import nltk
import re
import string
from nltk.tokenize import word_tokenize
from nltk.metrics.distance import edit_distance
from nltk.corpus import words

# Load the list of correct words
correct_words = set(words.words())

# Function to find the closest correct word
def find_closest_word(word):
    min_distance = float('inf')
    closest_word = None
    for correct_word in correct_words:
        distance = edit_distance(word, correct_word)
        if distance < min_distance:
            min_distance = distance
            closest_word = correct_word
    return closest_word

# Function to implement autocorrect for a sentence
def autocorrect_sentence(sentence):
    corrected_sentence = []
    # Tokenize the input sentence into words
    words = word_tokenize(sentence)
    for word in words:
        # Check if the word is alphabetic
        if word.isalpha():
            # Check if the word is in the list of correct words
            if word.lower() not in correct_words:
                # Find the closest correct word
                closest_word = find_closest_word(word.lower())
                # Preserve the original capitalization
                if word[0].isupper():
                    closest_word = closest_word.capitalize()
                corrected_sentence.append(closest_word)
            else:
                corrected_sentence.append(word)
        else:
            # Handle non-alphabetic tokens (like punctuation)
            corrected_sentence.append(word)
    return ' '.join(corrected_sentence)

# Example usage
input_sentence = "I AM DaCING."
corrected_sentence = autocorrect_sentence(input_sentence)
print("Original sentence:", input_sentence)
print("Corrected sentence:", corrected_sentence)
