import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report

# Cargar el modelo de spacy para lematización en español
nlp = spacy.load("es_core_news_sm")

# Función para lematizar el texto
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Cargar los datos
data = pd.read_csv('SMM4H_2024_Task3_Training_1800.csv', nrows=1800, usecols=[1, 2, 3], engine='python')

# Reemplazar los valores de la columna 'label' que no son 0 por 1
data['label'] = data['label'].apply(lambda x: 1 if x != 0 else x)

# Lematizar las columnas keyword y text
data['text'] = data['text'].apply(lemmatize_text)
data['keyword'] = data['keyword'].apply(lemmatize_text)

# Partición de los datos en train, validation y test
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Preprocesamiento de texto
vectorizer = CountVectorizer(binary=True)  # Usar binary=True para representación Bernoulli
X_train = vectorizer.fit_transform(train_data['keyword'])
X_val = vectorizer.transform(val_data['keyword'])
X_test = vectorizer.transform(test_data['keyword'])

y_train = train_data['label']
y_val = val_data['label']
y_test = test_data['label']

# Entrenar el modelo de Naive Bayes con distribución Bernoulli
model = BernoulliNB()
model.fit(X_train, y_train)

# Predicción en los datos de validación
predicted_labels = model.predict(X_val)

# Evaluación del modelo
print("Resultados en datos de validación:")
print(classification_report(y_val, predicted_labels))

# Prueba del modelo
X_test = vectorizer.transform(test_data['keyword'])
y_test = test_data['label']
predicted_labels_test = model.predict(X_test)

# Evaluación del modelo en los datos de prueba
print("\nResultados en datos de prueba:")
print(classification_report(y_test, predicted_labels_test))
