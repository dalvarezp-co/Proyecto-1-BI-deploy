#!/usr/bin/env python
# coding: utf-8

# In[40]:


#--------------Authors----------------
# t.acosta
# da.rubioh
# d.alvarezp


# In[41]:


import nltk


# In[42]:


nltk.download('stopwords')
nltk.download('stopwords')
nltk.download('wordnet')


# In[43]:


# Instalación de librerias
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import sys
from pandas_profiling import ProfileReport
from sklearn.naive_bayes import MultinomialNB
import re, string, unicodedata
import contractions
import inflect
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix
import scikitplot as skplt
from sklearn.base import BaseEstimator, ClassifierMixin
import unicodedata
import re
from inflect import engine
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt


# In[44]:


data=pd.read_excel('cat_345.xlsx')


# In[45]:


train=data


# In[46]:


train


# In[47]:


textos = train.copy()
textos['Conteo'] = [len(x) for x in textos['Textos_espanol']]
textos['Moda'] = [max(set(x.split(' ')), key = x.split(' ').count) for x in textos['Textos_espanol']]
textos['Max'] = [[max([len(x) for x in i.split(' ')])][0] for i in textos['Textos_espanol']]
textos['Min'] = [[min([len(x) for x in i.split(' ')])][0] for i in textos['Textos_espanol']]

# Se realiza un perfilamiento de los datos con la libre pandas profiling
ProfileReport(textos)


# # Preparación de datos

# * Limpieza de los datos.
# * Tokenización.
# * Normalización.
# * Vectorizacion

# In[48]:


from langdetect import detect


# In[49]:


def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('spanish'):
            new_words.append(word)
    return new_words

def remove_specialCoders(words):
    new_words = []
    for word in words:
        if "ao" in word:
            new_word = re.sub(r'ao', 'ú', word)
            new_words.append(new_word)
        elif "a3" in word:
            new_word = re.sub(r'a3', 'ó', word)
            new_words.append(new_word)
        elif "a3" in word:
            new_word = re.sub(r'Ã¡', 'á', word)
            new_words.append(new_word)
        elif "a3" in word:
            new_word = re.sub(r'Ã©', 'é', word)
            new_words.append(new_word)
        elif "a3" in word:
            new_word = re.sub(r'Ã', 'í', word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def replace_numbers(words, p):
    """Replace all integer occurrences in a list of tokenized words with textual representation"""
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def preprocessing(words):
    p = engine()
    words = to_lowercase(words)
    words = replace_numbers(words, p)
    words = remove_punctuation(words)
    words = remove_non_ascii(words)
    words = remove_stopwords(words)
    words = remove_specialCoders(words)
    return words


# ## **3.2 Tokenización**
# La tokenización permite dividir frases u oraciones en palabras. Con el fin de desglozar las palabras correctamente para el posterior análisis. Pero primero, se realiza una corrección de las contracciones que pueden estar presentes en los textos. 

# In[50]:


train['Textos_espanol'] = train['Textos_espanol'].apply(contractions.fix) #Aplica la corrección de las contracciones


# In[51]:


train['words'] = train['Textos_espanol'].apply(word_tokenize).apply(preprocessing) #Aplica la eliminación del ruido
train.head()


# In[52]:


train.head()


# #### **3.3. Normalización**
# En la normalización de los datos se realiza la eliminación de prefijos y sufijos, además de realizar una lemmatización.

# In[53]:


def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def stem_and_lemmatize(words):
    stems = stem_words(words)
    lemmas = lemmatize_verbs(words)
    return stems + lemmas

train['words'] = train['words'].apply(stem_and_lemmatize) #Aplica lematización y Eliminación de Prefijos y Sufijos.
train.head()


# ##### **3.4 Selección de campos**
# 
# Primero, se separa la variable predictora y los textos que se van a utilizar.

# In[54]:


train['words'] = train['words'].apply(lambda x: ' '.join(map(str, x)))
train


# In[55]:


X_datab, y_datab = train['words'],train['sdg']
y_datab


# In[56]:


X_datab


# In[57]:


count = CountVectorizer()
X_countb = count.fit_transform(X_datab)

print(X_countb.shape)
print(X_countb.toarray()[0])


# In[58]:


type(X_countb)


# In[59]:


X_countb.shape


# In[60]:


count = CountVectorizer(max_features=3000)  # Limita el vocabulario a las 1000 palabras más frecuentes
X_countb = count.fit_transform(X_datab)


# In[61]:


X_countb


# In[62]:


vocabulario = count.get_feature_names_out()
vocabulario


# In[63]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)


# In[64]:


X_trainb, X_testb, y_trainb, y_testb = train_test_split(X_countb, y_datab, test_size=0.2, random_state=42)


# In[65]:


model.fit(X_trainb, y_trainb)


# In[66]:


accuracy1 = model.score(X_testb, y_testb)
print("Exactitud del modelo:", accuracy1)


# # Creacion Modelo

# In[67]:


from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin

class TextPreprocessor2(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        p = engine()
        preprocessed_data = []
        
        for sentence in X:
            words = sentence.split()
            words = to_lowercase(words)
            words = replace_numbers(words, p)
            words = remove_punctuation(words)
            words = remove_non_ascii(words)
            words = remove_stopwords(words)
            words = remove_specialCoders(words)
            preprocessed_sentence = ' '.join(words)
            preprocessed_data.append(preprocessed_sentence)
        
        return pd.Series(preprocessed_data)

# Crear un pipeline
preprocessing_pipeline = Pipeline([
    ('text_preprocessor', TextPreprocessor2()),
    ('vectorizer',CountVectorizer(max_features=3000))
])


# In[68]:


x = train['Textos_espanol']
y = train['sdg']


# In[69]:


test = pd.read_excel('SinEtiquetatest_cat_345.xlsx')  # Asegúrate de proporcionar la ruta correcta


# In[70]:


x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
x_REAL_test = test["Textos_espanol"]


# In[71]:


x_REAL_test


# In[72]:


x_train.shape, y_train.shape    


# In[73]:


x_test.shape, y_test.shape    


# In[74]:


preprocessing_pipeline.fit(x_train)


# In[75]:


# Aplicar el pipeline a tus datos
preprocessed_train_data = preprocessing_pipeline.transform(x_train)


# In[76]:


type(preprocessed_train_data)


# In[77]:


preprocessed_train_data


# In[78]:


vocabulario = preprocessing_pipeline.named_steps["vectorizer"].get_feature_names_out()
vocabulario


# # Regresion Logistica Entrenamiento

# In[79]:


model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)


# In[80]:


X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)


# In[81]:


model.fit(X_trainb, y_trainb)


# In[82]:


accuracy2 = model.score(X_testb, y_testb)
print("Exactitud del modelo Regresion Logistica:", accuracy2)


# In[83]:


X_testb


# In[84]:


y_testb


# In[85]:


y_pred = model.predict(X_testb)


# In[86]:


cm = confusion_matrix(y_testb, y_pred)


# In[87]:


plt.figure(figsize=(18, 6)) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.show()


# In[88]:


plt.savefig("confusion_matrix_logist.png")


# # Transformacion Pipeline Test

# In[89]:


x_REAL_test


# In[90]:


preprocessed_REAL_test = preprocessing_pipeline.fit(x_REAL_test)


# In[91]:


preprocessed_REAL_test


# In[92]:


preprocessed_REAL_test = preprocessing_pipeline.transform(x_REAL_test)


# In[93]:


preprocessed_REAL_test


# # Predicciones Modelo Test Regresion Logistica

# In[94]:


model


# In[95]:


predictions = model.predict(preprocessed_REAL_test)


# In[96]:


predictions


# In[97]:


dataFramePredictedLogit = pd.DataFrame({"Textos_espanol": x_REAL_test, "Predictions": predictions})

# Visualiza el DataFrame
dataFramePredictedLogit


# # Naive Bayes Entrenamiento
# 

# In[98]:


from sklearn.naive_bayes import MultinomialNB


# In[99]:


X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)


# In[100]:


nb_classifier = MultinomialNB()


# In[101]:


nb_classifier.fit(X_trainb, y_trainb)


# In[102]:


accuracy3 = nb_classifier.score(X_testb, y_testb)
print("Exactitud del modelo Naive Bayes:", accuracy3)


# In[103]:


y_pred = nb_classifier.predict(X_testb)


# In[104]:


cm = confusion_matrix(y_testb, y_pred)
plt.figure(figsize=(18, 6)) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.show()
plt.savefig("confusion_matrix_Bayes.png")


# # Predicciones Bayes

# In[105]:


predictions = nb_classifier.predict(preprocessed_REAL_test)
dataFramePredictedBayes = pd.DataFrame({"Textos_espanol": x_REAL_test, "Predictions": predictions})
dataFramePredictedBayes


# # Random Forest Entrenamiento

# In[106]:


from sklearn.ensemble import RandomForestClassifier


# In[107]:


rf_classifier = RandomForestClassifier(n_estimators=200, random_state=42)


# In[108]:


X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)


# In[109]:


rf_classifier.fit(X_trainb, y_trainb)


# In[110]:


accuracy4 = rf_classifier.score(X_testb, y_testb)
print("Exactitud del modelo Random Forest:", accuracy4)


# In[111]:


y_pred = rf_classifier.predict(X_testb)


# In[112]:


cm = confusion_matrix(y_testb, y_pred)
plt.figure(figsize=(18, 6)) 
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)

plt.show()
plt.savefig("confusion_matrix_RandomForest.png")


# # Predicciones Random Forest

# In[113]:


predictions = rf_classifier.predict(preprocessed_REAL_test)
dataFramePredictedForest = pd.DataFrame({"Textos_espanol": x_REAL_test, "Predictions": predictions})
dataFramePredictedForest


# # KNN

# In[114]:


from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=300)
X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)
knn_classifier.fit(X_trainb, y_trainb)
accuracy = knn_classifier.score(X_testb, y_testb)
print("Exactitud del modelo knn_classifier:", accuracy)


# # svc

# In[115]:


from sklearn.svm import SVC
svm_classifier  = SVC(kernel='linear')
X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)
svm_classifier .fit(X_trainb, y_trainb)
accuracy = svm_classifier .score(X_testb, y_testb)
print("Exactitud del modelo svm_classifier :", accuracy)


# # GB classifier

# In[116]:


from sklearn.ensemble import GradientBoostingClassifier
gb_classifier = GradientBoostingClassifier(n_estimators=200, random_state=42)  # Puedes ajustar el número de estimadores según tus necesidades
X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)
gb_classifier .fit(X_trainb, y_trainb)
accuracy = gb_classifier .score(X_testb, y_testb)
print("Exactitud del modelo gb_classifier :", accuracy)


# # Bayes Bernulli

# In[117]:


from sklearn.naive_bayes import BernoulliNB
bnb_classifier = BernoulliNB()
X_trainb, X_testb, y_trainb, y_testb = train_test_split(preprocessed_train_data, y_train, test_size=0.2, random_state=42)
bnb_classifier.fit(X_trainb, y_trainb)
accuracy = bnb_classifier.score(X_testb, y_testb)
print("Exactitud del modelo bnb_classifier :", accuracy)


# # Eleccion Modelo Regresion

# In[118]:


dataFramePredictedBayes


# In[119]:


nombre_archivo = 'PrediccionesBayes.csv'  # Nombre del archivo CSV
dataFramePredictedBayes.to_csv(nombre_archivo, index=False)

