import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, LabelEncoder, MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes

# Configuração do tema do Seaborn
sns.set_theme(
    context="talk",
    style="darkgrid",
    rc={
        'figure.figsize': (8, 4),
    },
)

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
    
    missing_columns = [col for col in required_columns if col not in vendas.columns]
    if missing_columns:
        st.error(f"Colunas necessárias para o gráfico 3D não estão presentes nos dados: {', '.join(missing_columns)}")
        return
    
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
        
    st.write("Colunas disponíveis:", acessos.columns)

# Função para a página de análises de vendas
def pagina_vendas(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        # Garantir que 'Quantidade_de_Acessos' esteja presente
        if 'Quantidade_de_Acessos' not in vendas.columns:
            if 'Acessos' in vendas.columns:
                vendas['Quantidade_de_Acessos'] = vendas['Acessos']
            else:
                vendas['Quantidade_de_Acessos'] = 0

        # Criar uma coluna de devoluções
        vendas['Devolucoes'] = vendas.apply(lambda row: 'S' if pd.isna(row['N_Produtos']) or row['N_Produtos'] <= 0 else 'N', axis=1)

        # Seções de Análise
        st.title("Análise de Vendas")

        # Tabela de Estatísticas Resumidas
        st.header("Estatísticas Resumidas")
        estatisticas = pd.DataFrame({
            'Métrica': ['Ticket Médio', 'Faturamento Bruto', 'Faturamento Líquido', 'Total de Descontos', 'Total de Produtos Vendidos'],
            'Valor': [
                f"R${vendas['Vlr_Bruto'].sum() / vendas.shape[0]:.2f}",
                f"R${vendas['Vlr_Bruto'].sum():.2f}",
                f"R${vendas['Vlr_Liquido'].sum():.2f}",
                f"R${vendas['Vlr_Desconto'].sum():.2f}",
                f"{vendas['N_Produtos'].sum():.0f} unidades"
            ]
        })
        st.dataframe(estatisticas, use_container_width=True)

        # Top 10 Funcionários que Mais Venderam
        st.header("Top 10 Funcionários que Mais Venderam")
        func_rank = vendas['Funcionario'].value_counts().head(10)
        st.bar_chart(func_rank, use_container_width=True)

        # Funcionários com Menos Vendas
        st.header("Funcionários com Menos Vendas")
        func_rank_tail = vendas['Funcionario'].value_counts().tail(10)
        st.bar_chart(func_rank_tail, use_container_width=True)

        # Devoluções
        st.header("Devoluções")
        devolucao = vendas['Devolucoes'].value_counts()
        st.write(devolucao)

        # Visualizações detalhadas
        st.header("Visualizações Detalhadas")

        # Distribuição do Faturamento Líquido
        st.subheader("Distribuição do Faturamento Líquido")
        plt.figure()
        sns.histplot(vendas['Vlr_Liquido'], kde=True)
        plt.title('Distribuição do Faturamento Líquido')
        plt.xlabel('Faturamento Líquido')
        plt.ylabel('Frequência')
        st.pyplot(plt)

        # Correlação entre Variáveis
        st.subheader("Correlação entre Variáveis")
        plt.figure()
        corr = vendas[['Vlr_Bruto', 'Vlr_Desconto', 'Vlr_Liquido', 'N_Produtos']].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de Correlação')
        st.pyplot(plt)
        
    else:
        st.error("Dados de vendas não encontrados.")

# Função para a página de análise de clustering
def pagina_clustering(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        if 'Quantidade_de_Acessos' not in vendas.columns:
            if 'Acessos' in vendas.columns:
                vendas['Quantidade_de_Acessos'] = vendas['Acessos']
            else:
                vendas['Quantidade_de_Acessos'] = 0

        if 'Devolucoes' in vendas.columns:
            vendas['Devolucoes'] = vendas['Devolucoes'].map({'S': 1, 'N': 0})
        if 'Treinamento' in vendas.columns:
            vendas['Treinamento'] = vendas['Treinamento'].fillna('N').map({'S': 1, 'N': 0})
    

        vendas = vendas.select_dtypes(include=[np.number])
        vendas = vendas.fillna(0)

        numerical_columns = ['N_Produtos', 'Vlr_Bruto', 'Vlr_Liquido', 'Vlr_Desconto']
        vendas[numerical_columns] = vendas[numerical_columns].apply(scale)

        st.subheader("Métricas de Clustering")
        cluster_metrics = calcular_metricas_cluster(vendas)
        st.write(cluster_metrics)

        st.subheader("Aplicação do KMeans")
        kmeans = KMeans(n_clusters=4, random_state=0)
        kmeans_labels = kmeans.fit_predict(vendas)
        
        st.subheader("Gráfico 3D dos Clusters")
        plotar_grafico_3d(vendas, kmeans_labels)

    else:
        st.error("Dados de vendas não encontrados.")

# Função para a página de análise de regressão
def pagina_regressao(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        # Definindo a fórmula para o modelo de regressão linear
        function = 'Vlr_Liquido ~ Vlr_Bruto + Vlr_Desconto + N_Boletos + N_Produtos - 1'
        
        # Ajustando o modelo de regressão linear com statsmodels
        model = smf.ols(formula=function, data=vendas).fit()
        
        # Exibindo o resumo do modelo
        st.subheader("Resumo da Regressão Linear")
        st.write(model.summary())
        
        # Preparando os dados para a regressão linear
        x = vendas[['Vlr_Bruto', 'Vlr_Desconto', 'N_Boletos', 'N_Produtos']]
        y = vendas['Vlr_Liquido']
        
        # Aplicando Min-Max Scaling
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        
        # Separando os dados de treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)
        
        # Criando o objeto de Regressão Linear
        lr = LinearRegression()
        
        # Treinando o modelo
        lr.fit(x_train, y_train)
        
        # Calculando o coeficiente de determinação (R²) para os dados de treino e teste
        r_sq_train = lr.score(x_train, y_train)
        r_sq_test = lr.score(x_test, y_test)
        
        # Exibindo o coeficiente de determinação
        st.write(f'Coeficiente de Determinação (R²) - Treinamento: {r_sq_train:.2f}')
        st.write(f'Coeficiente de Determinação (R²) - Teste: {r_sq_test:.2f}')
        
        # Previsões e métricas de erro para dados de treinamento
        y_pred_train = lr.predict(x_train)
        st.write(f'MAE (Treinamento): {metrics.mean_absolute_error(y_train, y_pred_train):.2f}')
        st.write(f'MSE (Treinamento): {metrics.mean_squared_error(y_train, y_pred_train):.2f}')
        st.write(f'RMSE (Treinamento): {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f}')
        
        # Previsões e métricas de erro para dados de teste
        y_pred_test = lr.predict(x_test)
        st.write(f'MAE (Teste): {metrics.mean_absolute_error(y_test, y_pred_test):.2f}')
        st.write(f'MSE (Teste): {metrics.mean_squared_error(y_test, y_pred_test):.2f}')
        st.write(f'RMSE (Teste): {np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)):.2f}')
        
        # Visualização das previsões vs valores reais
        st.subheader("Visualização das Previsões vs Valores Reais")
        plt.figure()
        plt.scatter(y_test, y_pred_test, alpha=0.3)
        plt.xlabel('Valores Reais')
        plt.ylabel('Previsões')
        plt.title('Previsões vs Valores Reais')
        plt.grid(True)
        st.pyplot(plt)
        
        # Pairplot das variáveis
        st.subheader("Pairplot")
        sns_plot = sns.pairplot(vendas[['Vlr_Bruto', 'Vlr_Desconto', 'Vlr_Liquido', 'N_Produtos']])
        st.pyplot(sns_plot.figure)
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
                page = st.sidebar.selectbox("Selecione a página:", ["Análise de Acessos", "Análise de Vendas", "Clustering", "Regressão Linear"])
                
                if page == "Análise de Acessos":
                    pagina_acessos(dados)
                elif page == "Análise de Vendas":
                    pagina_vendas(dados)
                elif page == "Clustering":
                    pagina_clustering(dados)
                elif page == "Regressão Linear":
                    pagina_regressao(dados)

            else:
                st.warning("Não há pastas no diretório especificado.")
                
        except FileNotFoundError:
            st.error("O caminho fornecido não foi encontrado.")
        except Exception as e:
            st.error(f"Ocorreu um erro: {e}")

# Executar o aplicativo
if __name__ == "__main__":
    main()
