import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_generation import gerar_dados_realistas 

# Configurações gerais
st.set_page_config(page_title='Modelo Preditivo - ROI', layout='wide')

# Título do aplicativo
st.title('Modelo Preditivo - ROI')

# Carregar o dataset
st.header('1. Carregar e Preparar os Dados')

# Carregar o dataset
dataset = gerar_dados_realistas(10000)

# Exibir os primeiros dados
st.write(dataset.head())

# Codificar o canal de marketing
le = LabelEncoder()
dataset['canal_marketing'] = le.fit_transform(dataset['canal_marketing'])
dataset = dataset.drop(columns=['nome', 'titulo'])
dataset['budget_per_conversion'] = dataset['budget_total'] / (dataset['numero_conversoes'] + 1)

# Preparar os dados para modelagem
X = dataset[['faixa_etaria', 'budget_total', 'expectativa_alcance', 'canal_marketing', 'budget_per_conversion']]
y = dataset['roi']

# Escalar os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Dividir o dataset em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Treinar o modelo RandomForestRegressor
model_roi = RandomForestRegressor(n_estimators=100, random_state=42)
model_roi.fit(X_train, y_train)

# Avaliar o modelo
y_pred_roi = model_roi.predict(X_test)
mse_roi = mean_squared_error(y_test, y_pred_roi)
r2_roi = r2_score(y_test, y_pred_roi)

# Exibir métricas do modelo
st.subheader('1.1 Avaliação do Modelo')
st.write(f"Mean Squared Error: {mse_roi:.4f}")
st.write(f"R^2 Score: {r2_roi:.4f}")

# Coletar entrada do usuário para previsão
st.header('2. Fazer Previsões')

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
    min_value=0.0, 
    max_value=50000.0,  
    step=100.0  
)

expectativa_alcance_input = st.number_input(
    'Insira a Expectativa de Alcance', 
    min_value=0,  
    step=10  
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
roi_previsto = model_roi.predict(input_data_scaled)

# Exibir o resultado da previsão
st.subheader('Resultado da Previsão')
st.write(f"O ROI previsto é: R$ {roi_previsto[0]:.2f}")
