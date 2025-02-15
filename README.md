# Análise de Regressão Linear para Previsão de Vendas

Este projeto realiza uma análise de regressão linear para prever as vendas com base nos investimentos em três plataformas de marketing: YouTube, Facebook e Newspaper. O código foi desenvolvido em Python e utiliza bibliotecas como Pandas, NumPy, Seaborn, Matplotlib e Scikit-learn.

---

## Etapas do Projeto

### 1. Carregamento dos Dados
- Os dados são carregados a partir de um arquivo CSV (`MKT.csv`).
- Uma nova coluna `total` é criada, somando os investimentos nas três plataformas.

### 2. Tratamento dos Dados
- Verificação de valores nulos com `getData.count().isnull()`.
- Exibição de informações gerais sobre os dados com `getData.info()`.
- Visualização inicial dos dados com `getData.head()`.
- Cálculo de estatísticas descritivas com `getData.describe()`.

### 3. Divisão dos Dados
- Os dados são divididos em conjuntos de treino e teste (70% para treino e 30% para teste).
- Variáveis independentes (`X`): Investimentos nas plataformas (`youtube`, `facebook`, `newspaper`).
- Variável dependente (`Y`): Vendas (`sales`).

### 4. Treinamento do Modelo
- Um modelo de regressão linear é criado e treinado com os dados de treino (`x_train`, `y_train`).

### 5. Previsões
- O modelo é usado para prever as vendas com base nos dados de teste (`x_test`).

### 6. Avaliação do Modelo
- O desempenho do modelo é avaliado usando:
  - **Erro Quadrático Médio (MSE)**: Mede a média dos erros ao quadrado entre os valores reais e as previsões.
  - **Coeficiente de Determinação (R²)**: Indica a proporção da variabilidade dos dados que é explicada pelo modelo.
- Os valores de MSE e R² são exibidos no console.

### 7. Visualização dos Resultados
- **Comparação entre Valores Reais e Previsões**:
  - Gráfico de dispersão com uma linha ideal (diagonal) para comparação.
- **Impacto do Investimento nas Vendas**:
  - Gráficos de linha mostrando como o investimento em cada plataforma (YouTube, Facebook, Newspaper) afeta as vendas previstas.

---

## Como Executar

1. Certifique-se de que o arquivo `MKT.csv` está no caminho correto.
2. Execute o código em um ambiente Python com as bibliotecas necessárias instaladas.
3. Os resultados serão exibidos em gráficos e métricas impressas no console.

### Dependências
Instale as bibliotecas necessárias com o seguinte comando:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn
