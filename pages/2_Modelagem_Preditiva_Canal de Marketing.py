import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Importar os dados gerados de outro módulo
from data_generation import gerar_dados_realistas  # Certifique-se de que essa função já gera o DataFrame com os dados.

# Configurações gerais
st.set_page_config(page_title='Modelagem Preditiva - Canal de Marketing', layout='wide')

# Título do aplicativo
st.title('Modelagem Preditiva - Canal de Marketing')

# ================================================
# 1. Carregar os Dados
# ================================================

st.header('1. Carregar Dados')

# Gerar os dados realistas
df = gerar_dados_realistas(10000)

# Exibir os primeiros dados
st.write(df.head())

# ================================================
# 2. Preparar os Dados para Modelagem
# ================================================

# Selecionar as colunas relevantes
X = df[['faixa_etaria', 'budget_total', 'expectativa_alcance']]
y = df['canal_marketing']

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================================
# 3. Treinar o Modelo de Machine Learning
# ================================================

st.header('2. Treinar o Modelo de Machine Learning')

# Treinar o modelo Random Forest
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
accuracy_percentage = accuracy * 100
st.write(f"Acurácia do modelo: {accuracy_percentage:.2f}%")

# ================================================
# 4. Fazer Previsões com o Modelo
# ================================================

st.header('3. Fazer Previsões com o Modelo')

# Coletar entrada do usuário para previsão
faixa_etaria_opcoes = {
    'LIVRE PARA TODOS OS PÚBLICOS': 0,
    'MAIORES DE 10 ANOS': 10,
    'MAIORES DE 12 ANOS': 12,
    'MAIORES DE 14 ANOS': 14,
    'MAIORES DE 16 ANOS': 16,
    'MAIORES DE 18 ANOS': 18
}

faixa_etaria_input = st.selectbox(
    'Escolha a Faixa Etária', 
    options=list(faixa_etaria_opcoes.keys())
)

budget_input = st.number_input(
    'Budget Total', 
    min_value=0.0,  # Defina o valor mínimo para o orçamento
    max_value=500000.0,  # Defina o valor máximo para o orçamento
    step=100.0  # Incremento de 100 para o orçamento
)

expectativa_alcance_input = st.number_input(
    'Expectativa de Alcance', 
    min_value=0,  # Valor mínimo de expectativa de alcance
    step=10  # Incremento de 10 para a expectativa de alcance
)

# Preparar os dados de entrada
input_data = pd.DataFrame({
    'faixa_etaria': [faixa_etaria_opcoes[faixa_etaria_input]],  # Usar o valor correspondente à opção escolhida
    'budget_total': [budget_input],
    'expectativa_alcance': [expectativa_alcance_input]
})

# Fazer a previsão com o modelo treinado
prediction = model.predict(input_data)

# Exibir o resultado da previsão
st.subheader('Resultado da Previsão')
st.write(f'**Canal de Marketing Previsto:** {prediction[0]}')
