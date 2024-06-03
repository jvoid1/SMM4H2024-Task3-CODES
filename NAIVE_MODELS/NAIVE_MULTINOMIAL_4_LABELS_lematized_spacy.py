import pandas as pd
import spacy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

# Cargar el modelo de spacy para lematización en español
nlp = spacy.load("es_core_news_sm")

# Función para lematizar el texto
def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# Cargar los datos
data = pd.read_csv('SMM4H_2024_Task3_Training_1800.csv', nrows=1800, usecols=[1, 2, 3], engine='python')

# Lematizar las columnas keyword y text
data['text'] = data['text'].apply(lemmatize_text)
data['keyword'] = data['keyword'].apply(lemmatize_text)

# Partición de los datos en train, validation y test
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Preprocesamiento de texto y construcción del modelo
vectorizer = CountVectorizer()
model = MultinomialNB()

# Entrenar el modelo
X_train = vectorizer.fit_transform(train_data['text'])
y_train = train_data['label']
model.fit(X_train, y_train)

# Validación del modelo
X_val = vectorizer.transform(val_data['text'])
y_val = val_data['label']
predicted_labels_val = model.predict(X_val)

# Evaluación del modelo
print("Resultados en datos de validación:")
print(classification_report(y_val, predicted_labels_val))

# Calcular el micro-averaged F1 score
micro_f1_test = f1_score(y_val, predicted_labels_val, average='micro')

print("Micro-averaged F1 score:", micro_f1_test)

# Prueba del modelo
X_test = vectorizer.transform(test_data['text'])
y_test = test_data['label']
predicted_labels_test = model.predict(X_test)

# Evaluación del modelo en los datos de prueba
print("\nResultados en datos de prueba:")
print(classification_report(y_test, predicted_labels_test))

# Calcular el micro-averaged F1 score
micro_f1_test = f1_score(y_test, predicted_labels_test, average='micro')

print("Micro-averaged F1 score:", micro_f1_test)
