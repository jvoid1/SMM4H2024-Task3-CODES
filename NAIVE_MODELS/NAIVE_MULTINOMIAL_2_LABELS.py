import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report

# Cargar los datos
data = pd.read_csv('SMM4H_2024_Task3_Training_1800.csv', nrows=1800, usecols=[1, 2, 3])

# Reemplazar los valores de la columna 'label' que no son 0 por 1
data['label'] = data['label'].apply(lambda x: 1 if x != 0 else x)

# Partición de los datos en train, validation y test
train_data, temp_data = train_test_split(data, test_size=0.2, stratify=data['label'], random_state=42)
val_data, test_data = train_test_split(temp_data, test_size=0.5, stratify=temp_data['label'], random_state=42)

# Preprocesamiento de texto y construcción del modelo
vectorizer = TfidfVectorizer()
model = MultinomialNB()

# Entrenar el modelo
X_train = vectorizer.fit_transform(train_data['keyword'])
y_train = train_data['label']
model.fit(X_train, y_train)

# Validación del modelo
X_val = vectorizer.transform(val_data['keyword'])
y_val = val_data['label']
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
