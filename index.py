import streamlit as st # type: ignore
import pandas as pd # type: ignore
import numpy as np # type: ignore
import os 
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore
import plotly.express as px # type: ignore
import plotly.graph_objects as go # type: ignore
from sklearn.linear_model import LinearRegression # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn import metrics # type: ignore
from sklearn.preprocessing import LabelEncoder, scale # type: ignore
from sklearn.cluster import KMeans, AgglomerativeClustering # type: ignore
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score # type: ignore
from sklearn.mixture import GaussianMixture # type: ignore
from yellowbrick.cluster import kelbow_visualizer # type: ignore
from kmodes.kmodes import KModes # type: ignore
from kmodes.kprototypes import KPrototypes # type: ignore

# Função para carregar e tratar dados
def carregar_dados(caminho, pasta_selecionada):
    cliente = os.path.join(caminho, pasta_selecionada)
    dados = {}
    for csv in os.listdir(cliente):
        if csv.endswith('.csv'):
            if 'acessos' in csv:
                dados['acessos'] = pd.read_csv(os.path.join(cliente, csv))
            elif 'campanha' in csv:
                dados['campanha'] = pd.read_csv(os.path.join(cliente, csv))
            elif 'feedback' in csv:
                dados['feedback'] = pd.read_csv(os.path.join(cliente, csv))
            elif 'produto' in csv:
                dados['produto'] = pd.read_csv(os.path.join(cliente, csv))
            elif 'treinamento' in csv:
                dados['treinamento'] = pd.read_csv(os.path.join(cliente, csv))
            elif 'vendas' in csv:
                dados['vendas'] = pd.read_csv(os.path.join(cliente, csv))
    return dados

# Função para tratar dados nulos
def tratar_dados_nulos(df):
    df.replace('', pd.NA, inplace=True)
    for coluna in df.select_dtypes(include='number').columns:
        df[coluna].fillna(df[coluna].mean(), inplace=True)
    for coluna in df.select_dtypes(include='object').columns:
        df[coluna].fillna(df[coluna].mode()[0], inplace=True)
    return df

# Função para calcular métricas de clustering
def calcular_metricas_cluster(df):
    cluster_metrics = [silhouette_score, davies_bouldin_score, calinski_harabasz_score]
    resultados = []
    X = df.copy()

    for k in range(2, 11):
        model = KMeans(n_clusters=k, random_state=0)
        labels = model.fit_predict(X)
        resultados_dict = {'k': k, 'inertia': model.inertia_}
        for metric in cluster_metrics:
            resultados_dict[metric.__name__] = metric(X, labels)
        resultados.append(resultados_dict)

    return pd.DataFrame(resultados).set_index('k').style.background_gradient()

# Função principal do Streamlit
def main():
    st.title("Análise de Dados de Clientes")

    caminho = st.text_input("Digite o caminho do diretório:", r'/home/nickson/DNC-Projetos/Ser_Performance/Clientes')
    
    if caminho:
        try:
            # Obtendo as pastas dentro do diretório
            pastas = [p for p in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, p))]
            
            if pastas:
                pasta_selecionada = st.selectbox("Escolha uma pasta:", pastas)
                cliente = os.path.join(caminho, pasta_selecionada)

                # Carregar os dados da pasta selecionada
                dados = carregar_dados(caminho, pasta_selecionada)

                if 'acessos' in dados:
                    acessos = tratar_dados_nulos(dados['acessos'])
                    st.subheader("Acessos por Funcionário")
                    funcacessos = acessos.groupby('Funcionario')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
                    st.bar_chart(funcacessos.head(10))
                    
                    st.subheader("Acessos por Unidade")
                    unidade = acessos.groupby('Unidade')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
                    st.bar_chart(unidade.head(10))

                if 'vendas' in dados:
                    vendas = tratar_dados_nulos(dados['vendas'])
                    vendas['Devolucoes'] = vendas.apply(lambda row: 'S' if pd.isna(row['N_Produtos']) or row['N_Produtos'] <= 0 else 'N', axis=1)

                    st.subheader("Ticket Médio")
                    tcktmedio = vendas['Vlr_Bruto'].sum() / vendas.shape[0]
                    st.write(f'Ticket médio: R${tcktmedio:.2f}')

                    st.subheader("Faturamento Bruto")
                    FatBruto = vendas['Vlr_Bruto'].sum()
                    st.write(f'Faturamento bruto: R${FatBruto:.2f}')

                    st.subheader("Faturamento Líquido")
                    fatliq = vendas['Vlr_Liquido'].sum()
                    st.write(f'Faturamento líquido: R${fatliq:.2f}')

                    st.subheader("Total de Descontos")
                    total_descontos = vendas['Vlr_Desconto'].sum()
                    st.write(f'Total de descontos: R${total_descontos:.2f}')

                    st.subheader("Total de Produtos Vendidos")
                    totprod = vendas['N_Produtos'].sum()
                    st.write(f'Total de produtos: {totprod:.0f} unidades')

                    st.subheader("Top 10 Funcionários que Mais Venderam")
                    func_rank = vendas['Funcionario'].value_counts()
                    top10func = func_rank.head(10)
                    st.bar_chart(top10func)

                    st.subheader("Funcionários com Menos Vendas")
                    tail10func = func_rank.tail(10)
                    st.bar_chart(tail10func)

                    st.subheader("Devoluções")
                    devolucao = vendas['Devolucoes'].value_counts()
                    st.write(devolucao)
                    
                    st.write("Colunas disponíveis:", vendas.columns)

                    st.subheader("Análise de Clustering")

                    # Codificar variáveis categóricas e remover colunas não numéricas
                    if 'Devolucoes' in vendas.columns:
                        vendas['Devolucoes'] = vendas['Devolucoes'].map({'S': 1, 'N': 0})
                    if 'Treinamento' in vendas.columns:
                        vendas['Treinamento'] = vendas['Treinamento'].fillna('N').map({'S': 1, 'N': 0})

                    # Remover colunas não numéricas e colunas desnecessárias
                    vendas = vendas.select_dtypes(include=[np.number])
                    vendas = vendas.fillna(0)  # Preencher valores nulos

                    # Calcular métricas de cluster
                    cluster_metrics = calcular_metricas_cluster(vendas)
                    st.write(cluster_metrics)

                    # Aplicar KMeans
                    kmeans = KMeans(n_clusters=4, random_state=0)
                    kmeans_labels = kmeans.fit_predict(vendas)

                    # Gráfico 3D de clusters
                    fig = px.scatter_3d(
                        vendas,
                        x='N_Produtos',
                        y='Vlr_Liquido',
                        z='Quantidade_de_Acessos',
                        color=kmeans_labels.astype(str)
                    )
                    st.plotly_chart(fig)

                    # Análise de Regressão Linear
                    x = vendas[['Vlr_Bruto', 'Vlr_Desconto', 'N_Produtos', 'Funcionario', 'Quantidade_de_Acessos']]
                    y = vendas['Vlr_Liquido']
                    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
                    lr = LinearRegression()
                    lr.fit(x_train, y_train)
                    r_sq = lr.score(x, y)
                    st.write(f'Coeficiente de Determinação (R²): {r_sq:.2f}')
                    
                    y_pred_train = lr.predict(x_train)
                    st.write(f'MAE (Treinamento): {metrics.mean_absolute_error(y_train, y_pred_train):.2f}')
                    st.write(f'MSE (Treinamento): {metrics.mean_squared_error(y_train, y_pred_train):.2f}')
                    st.write(f'RMSE (Treinamento): {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f}')
                    
                    y_pred = lr.predict(x_test)
                    st.write(f'MAE (Teste): {metrics.mean_absolute_error(y_test, y_pred):.2f}')
                    st.write(f'MSE (Teste): {metrics.mean_squared_error(y_test, y_pred):.2f}')
                    st.write(f'RMSE (Teste): {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}')

            else:
                st.warning("Não há pastas no diretório especificado.")
                
        except FileNotFoundError:
            st.error("O caminho fornecido não foi encontrado.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

if __name__ == "__main__":
    main()
