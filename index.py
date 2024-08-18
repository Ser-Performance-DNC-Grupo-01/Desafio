import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

# Função para carregar e tratar dados
def carregar_dados(caminho, pasta_selecionada):
    cliente = os.path.join(caminho, pasta_selecionada)
    dados = {}
    for csv in os.listdir(cliente):
        if csv.endswith('.csv'):
            if 'acessos' in csv:
                dados['acessos'] = pd.read_csv(os.path.join(cliente, csv))
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
    
    # Reduzir dimensionalidade para clustering e visualização
    pca = PCA(n_components=min(df.shape[1], 5))
    df_reduzido = pca.fit_transform(df)
    
    for k in range(2, 6):
        model = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=0)
        labels = model.fit_predict(df_reduzido)
        resultados_dict = {'k': k, 'inertia': model.inertia_}
        for metric in cluster_metrics:
            resultados_dict[metric.__name__] = metric(df_reduzido, labels)
        resultados.append(resultados_dict)

    return pd.DataFrame(resultados).set_index('k').style.background_gradient()

# Função para plotar gráfico 3D de clusters
def plotar_grafico_3d(vendas, kmeans_labels):
    required_columns = ['N_Produtos', 'Vlr_Liquido', 'Quantidade_de_Acessos']
    
    # Verificar se todas as colunas necessárias estão presentes
    missing_columns = [col for col in required_columns if col not in vendas.columns]
    if missing_columns:
        st.error(f"Colunas necessárias para o gráfico 3D não estão presentes nos dados: {', '.join(missing_columns)}")
        return
    
    # Reduzir dimensionalidade para o gráfico 3D
    pca = PCA(n_components=3)
    vendas_reduzido = pca.fit_transform(vendas[required_columns])
    
    vendas_reduzido_df = pd.DataFrame(vendas_reduzido, columns=['PC1', 'PC2', 'PC3'])
    vendas_reduzido_df['Cluster'] = kmeans_labels.astype(str)
    
    fig = px.scatter_3d(
        vendas_reduzido_df,
        x='PC1',
        y='PC2',
        z='PC3',
        color='Cluster',
        labels={'PC1': 'Componente Principal 1', 'PC2': 'Componente Principal 2', 'PC3': 'Componente Principal 3'}
    )
    st.plotly_chart(fig)

# Função para a página de análises de acessos
def pagina_acessos(dados):
    if 'acessos' in dados:
        acessos = tratar_dados_nulos(dados['acessos'])
        st.subheader("Acessos por Funcionário")
        funcacessos = acessos.groupby('Funcionario')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
        st.bar_chart(funcacessos.head(10))
        
        st.subheader("Acessos por Unidade")
        unidade = acessos.groupby('Unidade')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
        st.bar_chart(unidade.head(10))
    else:
        st.error("Dados de acessos não encontrados.")

# Função para a página de análises de vendas
def pagina_vendas(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        # Criar a coluna 'Quantidade_de_Acessos' se não existir
        if 'Quantidade_de_Acessos' not in vendas.columns:
            if 'Acessos' in vendas.columns:
                vendas['Quantidade_de_Acessos'] = vendas['Acessos']
            else:
                vendas['Quantidade_de_Acessos'] = 0  # Preencher com zero se não há dados

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
    else:
        st.error("Dados de vendas não encontrados.")

# Função para a página de análise de clustering
def pagina_clustering(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        # Criar a coluna 'Quantidade_de_Acessos' se não existir
        if 'Quantidade_de_Acessos' not in vendas.columns:
            if 'Acessos' in vendas.columns:
                vendas['Quantidade_de_Acessos'] = vendas['Acessos']
            else:
                vendas['Quantidade_de_Acessos'] = 0  # Preencher com zero se não há dados

        # Codificar variáveis categóricas e remover colunas não numéricas
        if 'Devolucoes' in vendas.columns:
            vendas['Devolucoes'] = vendas['Devolucoes'].map({'S': 1, 'N': 0})
        if 'Treinamento' in vendas.columns:
            vendas['Treinamento'] = vendas['Treinamento'].fillna('N').map({'S': 1, 'N': 0})

        # Remover colunas não numéricas e colunas desnecessárias
        vendas = vendas.select_dtypes(include=[np.number])
        vendas = vendas.fillna(0)  # Preencher valores nulos

        # Calcular métricas de cluster
        st.subheader("Métricas de Clustering")
        cluster_metrics = calcular_metricas_cluster(vendas)
        st.write(cluster_metrics)

        # Aplicar KMeans
        st.subheader("Aplicação do KMeans")
        kmeans = KMeans(n_clusters=4, random_state=0)
        kmeans_labels = kmeans.fit_predict(vendas)
        
        # Plotar gráfico 3D de clusters
        st.subheader("Gráfico 3D dos Clusters")
        plotar_grafico_3d(vendas, kmeans_labels)

    else:
        st.error("Dados de vendas não encontrados.")

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

                # Configurar a navegação entre páginas
                page = st.sidebar.selectbox("Selecione a página:", ["Análise de Acessos", "Análise de Vendas", "Clustering"])
                
                if page == "Análise de Acessos":
                    pagina_acessos(dados)
                elif page == "Análise de Vendas":
                    pagina_vendas(dados)
                elif page == "Clustering":
                    pagina_clustering(dados)

            else:
                st.warning("Não há pastas no diretório especificado.")
                
        except FileNotFoundError:
            st.error("O caminho fornecido não foi encontrado.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Executar o aplicativo
if __name__ == "__main__":
    main()
