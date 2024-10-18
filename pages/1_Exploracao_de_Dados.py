import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assumindo que o dataset_realista foi importado do data_generation.py
from data_generation import gerar_dados_realistas

# Gerar o dataset
st.title("Exploração de Dados")

# Dataset gerado a partir do data_generation.py
dataset_realista = gerar_dados_realistas(10000)

# Exibir o dataset
st.write("Visualizando o dataset gerado:")
st.write(dataset_realista)

# Selecionar apenas as colunas numéricas para a correlação
colunas_numericas = dataset_realista.select_dtypes(include=['float64', 'int64'])

# Calcular a correlação
corr = colunas_numericas.corr()

# Exibir a matriz de correlação
st.write("Matriz de correlação:")
st.write(corr)

# Plotar um heatmap da matriz de correlação
st.write("Heatmap da matriz de correlação:")
plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1, center=0)
st.pyplot(plt)

# Descrever as estatísticas gerais do dataset
st.write("Estatísticas descritivas do dataset:")
st.write(dataset_realista.describe())
