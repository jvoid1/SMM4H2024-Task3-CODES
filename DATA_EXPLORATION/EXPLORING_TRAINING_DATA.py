import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
import re
from collections import Counter
from collections import defaultdict

def analyze_data(data):
    # Mostrar las clases y su recuento
    print("Recuento de clases:")
    class_counts = data['label'].value_counts()
    print(class_counts)
    
    # Graficar las clases
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar', color='skyblue')
    plt.title('Gráfico de Clases\n1 = positive effect, 2 = neutral or no effect, 3 = negative effect, or 0 = unrelated')
    plt.xlabel('Clases')
    plt.ylabel('Textos anotados con la clase')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

    # Preprocesamiento de texto
    data['text'] = data['text'].astype(str)
    data['keyword'] = data['keyword'].astype(str)
    data['word_count'] = data['text'].apply(lambda x: len(nltk.word_tokenize(x)))
    data['digit_count'] = data['text'].apply(lambda x: len(re.findall(r'\d', x)))

    # Estadísticas generales
    print("\nESTADÍSTICAS GENERALES:")
    print("Promedio de palabras por texto:", data['word_count'].mean())
    print("Longitud promedio de palabras:", average_word_length(data['text']))
    print("Promedio de palabras únicas por texto:", average_unique_words(data['text']))
    print("Promedio de números/dígitos por texto:", data['digit_count'].mean())
    pos_counts = average_pos_counts(data['text'])
    print("Promedio de verbos por texto:", pos_counts['verbs'])
    print("Promedio de sustantivos por texto:", pos_counts['nouns'])
    print("Promedio de adjetivos por texto:", pos_counts['adjectives'])
    text_length_distribution(data['text'])
    
    # Palabras más comunes
    print("\nPalabras más comunes:")
    print(get_top_words(data['text']))
    print("Palabras más comunes (sin stopwords):")
    print(get_top_words_without_stopwords(data['text']))

    # Keywords
    print("\nAnálisis de Keywords:")
    analyze_keywords(data['keyword'])

def average_word_length(text_column):
    all_words = ' '.join(text_column).split()
    total_characters = sum(len(word) for word in all_words)
    total_words = len(all_words)
    return total_characters / total_words

def average_unique_words(text_column):
    unique_word_counts = text_column.apply(lambda x: len(set(x.split())))
    return unique_word_counts.mean()

def average_pos_counts(text_column):
    pos_counts = defaultdict(int)
    total_texts = len(text_column)
    
    for text in text_column:
        tokens = word_tokenize(text)
        tagged_tokens = pos_tag(tokens)
        
        for token, pos in tagged_tokens:
            if pos.startswith('V'):
                pos_counts['verbs'] += 1
            elif pos.startswith('N'):
                pos_counts['nouns'] += 1
            elif pos.startswith('J'):
                pos_counts['adjectives'] += 1
    
    avg_pos_counts = {pos: count / total_texts for pos, count in pos_counts.items()}
    return avg_pos_counts

def text_length_distribution(text_column):
    text_lengths = [len(text.split()) for text in text_column]
    plt.figure(figsize=(10, 6))
    sns.histplot(text_lengths, bins=20, color='skyblue', kde=True)
    plt.title('Distribución de Longitud de Textos')
    plt.xlabel('Longitud del Texto')
    plt.ylabel('Frecuencia')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def get_top_words(text_column, n=10):
    all_words = ' '.join(text_column).split()
    word_counts = Counter(all_words)
    return word_counts.most_common(n)

def get_top_words_without_stopwords(text_column, n=10):
    stop_words = set(nltk.corpus.stopwords.words('english'))
    word_counter = Counter()
    for text in text_column:
        words = re.findall(r'\b\w+\b', text.lower())
        filtered_words = [word for word in words if word not in stop_words]
        word_counter.update(filtered_words)
    return word_counter.most_common(n)

def analyze_keywords(keyword_column):
    all_keywords = ', '.join(keyword_column).split(', ')
    keyword_counts = Counter(all_keywords)
    
    avg_keywords_per_text = len(all_keywords) / len(keyword_column)
    print("Promedio de keywords por texto:", avg_keywords_per_text)

    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
    print("Número de Keywords diferentes:", len(sorted_keywords))
    print("Frecuencia de Keywords:")
    for keyword, count in sorted_keywords:
        print(f'{keyword}: {count}')

    sorted_keywords = sorted_keywords[:20]
    keywords, counts = zip(*sorted_keywords)

    plt.figure(figsize=(10, 6))
    plt.bar(keywords, counts, color='skyblue')
    plt.title('Frecuencia de las Keywords')
    plt.xlabel('Keyword')
    plt.ylabel('Frecuencia')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

# Cargar los datos de entrenamiento
data = pd.read_csv('SMM4H_2024_Task3_Training_1800.csv', nrows=1801, usecols=[0, 1, 2, 3], engine='python')

# Preguntar al usuario por la clase o análisis completo
opcion = input("¿Deseas realizar el análisis en todo el dataset o solo en los datos de alguna clase específica? (Ingrese 'y' para análisis en todo el dataset o el número de la clase) \n(1 = positive effect, 2 = neutral or no effect, 3 = negative effect, or 0 = unrelated):")

if opcion == 'y':
    analyze_data(data)
elif opcion.isdigit() and int(opcion) in data['label'].unique():
    filtered_data = data[data['label'] == int(opcion)]
    analyze_data(filtered_data)
else:
    print("Opción no válida.")
