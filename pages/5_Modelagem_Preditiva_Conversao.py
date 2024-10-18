import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_generation import gerar_dados_realistas 

# Configurações gerais
st.set_page_config(page_title='Modelo Preditivo - Número de Conversões', layout='wide')

# Título do aplicativo
st.title('Modelo Preditivo - Número de Conversões')

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

# Codificar o canal de marketing
le = LabelEncoder()
df['canal_marketing'] = le.fit_transform(df['canal_marketing'])
df['budget_per_conversion'] = df['budget_total'] / (df['numero_conversoes'] + 1)

# Selecionar as colunas relevantes
X = df[['faixa_etaria', 'budget_total', 'expectativa_alcance', 'canal_marketing', 'budget_per_conversion']]
y = df['numero_conversoes']

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir o dataset em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# ================================================
# 3. Treinar o Modelo de Machine Learning
# ================================================

st.header('2. Treinar o Modelo de Machine Learning')

# Treinar o modelo RandomForestRegressor
model_numero_conversoes = RandomForestRegressor(n_estimators=100, random_state=42)
model_numero_conversoes.fit(X_train, y_train)

# Fazer previsões
y_pred_numero_conversoes = model_numero_conversoes.predict(X_test)

# Avaliar o modelo
mse_numero_conversoes = mean_squared_error(y_test, y_pred_numero_conversoes)
r2_numero_conversoes = r2_score(y_test, y_pred_numero_conversoes)

# Exibir métricas do modelo
st.subheader('Avaliação do Modelo')
st.write(f"Mean Squared Error: {mse_numero_conversoes:.4f}")
st.write(f"R^2 Score: {r2_numero_conversoes:.4f}")

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
    'Insira o Budget Total', 
    min_value=0.0,  # Defina o valor mínimo para o orçamento
    max_value=500000.0,  # Defina o valor máximo para o orçamento
    step=100.0  # Incremento de 100 para o orçamento
)

expectativa_alcance_input = st.number_input(
    'Insira a Expectativa de Alcance', 
    min_value=0,  # Valor mínimo de expectativa de alcance
    step=10  # Incremento de 10 para a expectativa de alcance
)

canal_marketing_input = st.selectbox(
    'Escolha o Canal de Marketing', 
    options=le.classes_  # Usar as classes do LabelEncoder
)

# Preparar os dados de entrada para previsão
input_data = pd.DataFrame({
    'faixa_etaria': [faixa_etaria_opcoes[faixa_etaria_input]],  # Usar o valor correspondente à opção escolhida
    'budget_total': [budget_input],
    'expectativa_alcance': [expectativa_alcance_input],
    'canal_marketing': [le.transform([canal_marketing_input])[0]],  # Codificar o canal de marketing
    'budget_per_conversion': [budget_input / (1 + 1)]  # Exemplo de cálculo
})

# Escalar os dados de entrada
input_data_scaled = scaler.transform(input_data)

# Fazer a previsão
numero_conversoes_previsto = model_numero_conversoes.predict(input_data_scaled)

# Exibir o resultado da previsão
st.subheader('Resultado da Previsão')
st.write(f"O número de conversões previsto é: {numero_conversoes_previsto[0]:.2f}")