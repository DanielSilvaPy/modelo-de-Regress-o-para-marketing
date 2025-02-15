# Tratamentos dos dados.
import pandas as pd
import numpy as np

# Visualização dos dados
import seaborn as sns
import matplotlib.pyplot as plt

# 01 Trazer os dados.
getData = pd.read_csv("C:/Users/danie/PycharmProjects/projectmkt/data/MKT.csv", sep=",")

# 02 Tratamento dos dados.
getData['total'] = getData['youtube'] + getData['facebook'] + getData['newspaper']

print(getData.count().isnull()) # Verificando se temos linhas nulas
print(getData.info())
print(getData.head())
print(getData.describe())

# 03 Dividir os dados em treino e teste - Um para treinamento e outro para teste.
from sklearn.model_selection import train_test_split

X = getData[['youtube', 'facebook', 'newspaper']] # Váriaves X são váriaves que tentatam explicar na avaliação da venda.
Y = getData['sales'] # Váriaves Y Tentar Prever.

x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size = 0.7, test_size = 0.3, random_state = 42)

# 04 Criar e treinar o modelo
from sklearn.linear_model import LinearRegression
modelo = LinearRegression().fit(x_train, y_train) # Treinar o modelo com os dados de treino.

# 05 Fazer previsões
y_pred = modelo.predict(x_test)

# 06. Avaliar o modelo. valie o desempenho do modelo utilizando métricas como Erro Quadrático Médio (MSE) e Coeficiente de Determinação (R²)
from sklearn.metrics import mean_squared_error
MSE = mean_squared_error(y_test, y_pred)
print(f'O valor do MSE é: {MSE}')

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_pred)
print(f'O valor de r2 é: {r2}')

# 07 Visualização dos dados.
plt.scatter(y_test, y_pred, color = 'blue',  label = 'Previsões vs. Reais')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal')
plt.xlabel('Valores Reais (y_test)')
plt.ylabel('Previsões (y_pred)')
plt.title('Comparação entre Valores Reais e Previsões')
plt.legend()
plt.show()

# Criar uma faixa de investimento para cada plataforma
investimento_range = np.linspace(0, getData[['youtube', 'facebook', 'newspaper']].max().max(), 100)

# Função para prever as vendas com base no investimento em uma plataforma

def prever_vendas_por_plataforma(plataforma, investimento_range):
    if plataforma == 'youtube':
        X_pred = pd.DataFrame({
            'youtube': investimento_range,
            'facebook': 0,
            'newspaper': 0
        })
    elif plataforma == 'facebook':
        X_pred = pd.DataFrame({
            'youtube': 0,
            'facebook': investimento_range,
            'newspaper': 0
        })
    elif plataforma == 'newspaper':
        X_pred = pd.DataFrame({
            'youtube': 0,
            'facebook': 0,
            'newspaper': investimento_range
        })
    return  modelo.predict(X_pred)

# Plotar gráficos de linha para cada plataforma
plt.figure(figsize=(12, 6))

# YouTube
plt.subplot(1, 3, 1)
plt.plot(investimento_range, prever_vendas_por_plataforma('youtube', investimento_range), color='blue')
plt.xlabel('Investimento no YouTube')
plt.ylabel('Vendas Previstas')
plt.title('Impacto do YouTube nas Vendas')

# Facebook
plt.subplot(1, 3, 2)
plt.plot(investimento_range, prever_vendas_por_plataforma('facebook', investimento_range), color='green')
plt.xlabel('Investimento no Facebook')
plt.ylabel('Vendas Previstas')
plt.title('Impacto do Facebook nas Vendas')

# Newspaper
plt.subplot(1, 3, 3)
plt.plot(investimento_range, prever_vendas_por_plataforma('newspaper', investimento_range), color='red')
plt.xlabel('Investimento no Newspaper')
plt.ylabel('Vendas Previstas')
plt.title('Impacto do Newspaper nas Vendas')

plt.tight_layout()
plt.show()