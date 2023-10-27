import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import proyecto_grupo13
from pandas_profiling import ProfileReport
import plotly.express as px

#profile = ProfileReport(proyecto_grupo13.textos)
#profile.to_file("profile_report.html")

# Carga los datos desde Jupyter Notebook
# Reemplaza 'data.csv' con la ruta de tu archivo de datos
data = pd.read_excel('cat_345.xlsx')

dataGraph = {
    'Modelo' : ['Regresión Logística', 'Naive Bayes', 'Random Forest'],
    'Exactitud' : [proyecto_grupo13.accuracy2, proyecto_grupo13.accuracy3, proyecto_grupo13.accuracy4]
}

df = pd.DataFrame(dataGraph)

# Título de la aplicación
st.title('Modelo Analítico Grupo 13')

# Muestra tus datos en una tabla
st.write('**Datos:**', data)

st.title('Preparación de Datos')
## Pandas profiling
#st.header('Data Profiling con pandas profiling')
#st.write("Información general sobre los datos:")
#with open("profile_report.html", 'r', encoding='utf-8') as file:
#    st.markdown(file.read())
#st.text('')
st.write('**Limpieza de datos:**')
st.markdown("""
    - Remover caracteres no ASCII
    - Pasar todos los caracteres a minúsculas
    - Remover signos de puntuación
    - Remover stop words
    - Reemplazar special coders por caracteres con tildes
    - Reemplazar números por su representación textual
    """)
st.text('')
st.write('**Tokenización:**')
st.write('La tokenización  permite dividir frases u oraciones en palabras. Con el fin de desglozar las palabras correctamente para el posterior análisis.')
st.text('')
st.write('**Normalización:**')
st.write('En la normalización de los datos se realiza la eliminación de prefijos y sufijos, además de realizar una lemmatización.')
st.write('Datos después de limpieza, tokenización, normalización y vectorización', proyecto_grupo13.train)

st.header('Exactitud del modelo')
st.write(proyecto_grupo13.accuracy1)

## Describir brevemente cada modelo y razones de elección
## Mostrar matrices de confusión creadas
st.title('Resultados')
st.write('_Nota: Las exactitudes son valores entre 0 y 1_')

## Gráficos sobre resultados
st.title('Gráfico comparativo sobre modelos')
fig = px.bar(df, x='Modelo', y='Exactitud', title='Gráfico de Barras')
st.plotly_chart(fig)

st.header('Regresión Logística Multivariada')
st.markdown('<div style="text-align: justify;">Este modelo nos permite analizar opiniones y relacionarlas con los objetivos de desarrollo sostenible (ODS), lo que es esencial para el trabajo de la UNFPA en el seguimiento, la evaluación de políticas públicas y su impacto social.</div>', unsafe_allow_html=True)
st.text('')
st.write('Exactitud del modelo de Regresión Logística:', proyecto_grupo13.accuracy2)
st.write('**Razones de elección:**')
st.markdown("""
    - Capacidad para manejar problemas de clasificación multiclase
    - Interpretabilidad
    - Suposiciones
    """)
st.subheader('Matriz de confusión - Regresión Logística')
st.image('./confusion_matrix_logist.png')
st.subheader('Predicciones Modelo Test Regresión Logística')
st.write(proyecto_grupo13.dataFramePredictedLogit)


st.header('Naive Bayes')
st.markdown('<div style="text-align: justify;">Este modelo ofrece una suposición de independencia entre las distintas variables predictoras, tiene predicciones probabilísticas, no es propenso al sobreajuste y por último, es usado en clasificación de textos como detección de spam, clasificación de documentos, entre otros.</div>', unsafe_allow_html=True)
st.text('')
st.write('Exactitud del modelo de Naive Bayes:', proyecto_grupo13.accuracy3)
st.write('**Razones de elección:**')
st.markdown("""
    - Convergencia rápida, puede recurrir menos datos de entrenamiento para alcanzar un rendimiento comparable
    - Simplicidad y eficiencia
    - Manejo de datos de alta dimensionalidad
    """)
st.subheader('Matriz de confusión - Naive Bayes')
st.image('./confusion_matrix_Bayes.png')
st.subheader('Predicciones Modelo Naive Bayes')
st.write(proyecto_grupo13.dataFramePredictedBayes)

st.header('Random Forest')
st.markdown('<div style="text-align: justify;">Este es un modelo que implementa un algoritmo de conjunto que combina múltiples árboles de decisión para mejorar la precisión, cada árbol en el bosque se entrena con una muestra de datos ligeramente diferente y utiliza una selección aleatoria de características para prevenir el sobreajuste.</div>', unsafe_allow_html=True)
st.text('')
st.write('Exactitud del modelo de Random Forest:', proyecto_grupo13.accuracy4)
st.write('**Razones de elección:**')
st.markdown("""
    - Versatilidad para uso de clasificación y regresión, en analítica de textos permite la detección de spam, análisis de sentimientos o clasificación de temas
    - Reduce el sobreajuste por la aleatoriedad en la construcción de árboles
    - Balance de clases ya que mitiga problemas de sesgo
    """)
st.subheader('Matriz de confusión - Random Forest')
st.image('./confusion_matrix_RandomForest.png')
st.subheader('Predicciones Modelo Random Forest')
st.write(proyecto_grupo13.dataFramePredictedForest)

st.title('Conclusión')
st.markdown('<div style="text-align: justify;">Se recomienda el uso del modelo de Naive Bayes, este obtuvo la exactitud más alta con un valor de 97.1%. No obstante, esta no es la única razón por la que se recomienda su uso sino también porque es fácilmente escalable, lo que ayuda a manejar cantidades más grandes de datos. Asimismo, su simplicidad para operar en este tipo de tareas de clasificación de textos.</div>', unsafe_allow_html=True)