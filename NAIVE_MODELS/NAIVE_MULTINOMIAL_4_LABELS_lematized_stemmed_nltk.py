import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Inicializar el stemmer en inglés
stemmer = SnowballStemmer('english')

# Función para realizar stemming en el texto
def stem_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Realizar stemming en cada token y excluir las stopwords
    stemmed_words = [stemmer.stem(word) for word in tokens if word.lower() not in stopwords.words('english')]
    return ' '.join(stemmed_words)

# Función para lematizar el texto con NLTK
def lemmatize_text(text):
    # Tokenizar el texto
    tokens = word_tokenize(text)
    # Obtener las palabras lematizadas excluyendo las stopwords
    lemmatized_words = [word for word in tokens if word.lower() not in stopwords.words('spanish')]
    return ' '.join(lemmatized_words)

# Cargar los datos
data = pd.read_csv('SMM4H_2024_Task3_Training_1800.csv', nrows=1800, usecols=[1, 2, 3], engine='python')

# Lematizar las columnas 'keyword' y 'text'
data['keyword'] = data['keyword'].apply(stem_text)
data['text'] = data['text'].apply(stem_text)

# Partición de los datos en train, validation y test
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Preprocesamiento de texto y construcción del modelo
vectorizer = TfidfVectorizer()
model = MultinomialNB()

# Entrenar el modelo
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']
model.fit(X_train, y_train)

# Validación del modelo
X_val = vectorizer.transform(val_data['text'])
y_val = val_data['label']
predicted_labels = model.predict(X_val)

# Evaluación del modelo
print("Resultados en datos de validación:")
print(classification_report(y_val, predicted_labels))

# Prueba del modelo
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']
predicted_labels_test = model.predict(X_test)

# Evaluación del modelo en los datos de prueba
print("\nResultados en datos de prueba:")
print(classification_report(y_test, predicted_labels_test))
