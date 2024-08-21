import streamlit as st
import pandas as pd
import numpy as np
import os
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, MeanShift, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import statsmodels.formula.api as smf
from PIL import Image
st.set_page_config(layout="wide")

with open("D:\PROJETO_REAL_DNC_-SER-PERFORMACE\git\style.css") as g:
    st.markdown(f"<style>{g.read()}</style>", unsafe_allow_html= True)


logo_path = r'D:\PROJETO_REAL_DNC_-SER-PERFORMACE\git\logo.png'  
logo = Image.open(logo_path)

st.image(logo, width=300)

st.markdown("""
   
    <style>
    
    .st-emotion-cache-1itdyc2.eczjsme18{
        
        box-shadow: 0 4px 8px rgb(241, 133, 10); 
    }
    .logo {
        margin-right: 20px;
    }
    h1{
        color: #ec9509;
        font-size: 50px;
        text-align: center;
    }
    </style>
    <div class="head">
        <h1>Análise de Dados de Clientes</h1>
    </div>
""", unsafe_allow_html=True)


col1, col2, col3, col4, col5, col6 = st.columns(6)


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
  
  
  
        # Criação das colunas
        col1, col2 = st.columns([1, 1])  # Define o espaço igual para as colunas

        #  gráficos de acessos
        with col1:
           
            st.subheader("Acessos por Funcionário")
            funcacessos = acessos.groupby('Funcionario')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
            fig1 = go.Figure(data=go.Bar(
                x=funcacessos.head(10).index, 
                y=funcacessos.head(10),
                marker_color='orange',
                text=[],  
                textposition='none'  
            ))
            fig1.update_layout(
                height=550,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                title='',
                title_font_color='orange',
                xaxis_title='',
                yaxis_title='Quantidade de Acessos',
                xaxis_title_font_color='orange',
                yaxis_title_font_color='orange',
                xaxis_tickfont_color='orange',
                yaxis_tickfont_color='orange',
                margin=dict(l=40, r=20, t=40, b=40),
                showlegend=False  # Remove a legenda
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            st.subheader("Acessos por Unidade")
            unidade = acessos.groupby('Unidade')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
            fig2 = go.Figure(data=go.Bar(x=unidade.head(10).index, y=unidade.head(10), marker_color='orange'))
            fig2.update_layout(
                height=550,
                paper_bgcolor='#0E1117',
                plot_bgcolor='#0E1117',
                title='',
                title_font_color='orange',
                xaxis_title='',
                yaxis_title='Quantidade de Acessos',
                xaxis_title_font_color='orange',
                yaxis_title_font_color='orange',
                xaxis_tickfont_color='orange',
                yaxis_tickfont_color='orange',
                margin=dict(l=40, r=20, t=40, b=40)
            )
            st.plotly_chart(fig2, use_container_width=True)

    else:
        st.error("Dados de acessos não encontrados.")

# Função para a página de análises de vendas

def pagina_vendas(dados):
    if 'vendas' in dados:
        vendas = tratar_dados_nulos(dados['vendas'])
        
        if 'Quantidade_de_Acessos' not in vendas.columns:
            if 'Acessos' in vendas.columns:
                vendas['Quantidade_de_Acessos'] = vendas['Acessos']
        
        col1, col2 = st.columns([1, 1])
        
        vendas['Devolucoes'] = vendas.apply(lambda row: 'S' if pd.isna(row['N_Produtos']) or row['N_Produtos'] <= 0 else 'N', axis=1)
        
        # Calcular valores
        tcktmedio = vendas['Vlr_Bruto'].sum() / vendas.shape[0]
        FatBruto = vendas['Vlr_Bruto'].sum()
        fatliq = vendas['Vlr_Liquido'].sum()
        totprod = vendas['N_Produtos'].sum()
        devolucao = vendas['Devolucoes'].value_counts()
        total_descontos = vendas['Vlr_Desconto'].sum()
        
        
        
        
        
            

           # st.subheader("Total de Produtos Vendidos")
        totprod = vendas['N_Produtos'].sum()
           # st.write(f'Total de produtos: {totprod:.0f} unidades')
         
         #barra_laranja   
        with col1:
                st.markdown(
                f"""
                <style>
                    .card {{
                    display: flex;
                    justify-content: center;
                    align-items: center; 
                    gap: 180px;        
                    width: 81vw;;
                    background-color: #FFA500; 
                    color: black;
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 50px;
                }}
                .PV .h1{{
                    color: #942f2f;
                }}
                .TM{{
                   
                    display: flex;
                    gap: 180px;  
                }} 
                              
                </style>
                <div class="card">
                    <div class="PV"
                    <h1 class="vendidos">Faturamento Bruto</h1><br>
                    R${FatBruto:.2f}
                    </div>                    
                    <div class="TM"
                    <h1>Ticket Médio</h1><br>
                    R${tcktmedio:.2f}
                    <div/>
                    <div class="TM"
                    <h1>Faturamento Líquido</h1><br>
                    R${fatliq:.2f}
                    <div/>
                    <div class="TM"
                    <h1>Produtos Vendidos</h1><br>
                    {totprod:.0f} 
                    <div/>
                    <div class="TM"
                    <h1>Total de Descontos</h1><br>
                    {total_descontos:.2f} 
                    <div/>                          
                                         
                   
                </div>
                             
                """,
                unsafe_allow_html=True
            )
            

        
        
        
     #Gráfico de vendas
        col10, col20 = st.columns(2)    
        with col10:  
            st.subheader("Top 10 Funcionários que Mais Venderam")
            func_rank = vendas['Funcionario'].value_counts()
        # top10func = func_rank.head(10)
    with col10:
        fig3 = go.Figure(data=go.Bar(
            x=func_rank.head(10).index, 
            y=func_rank.head(10),
            marker_color='orange',
            text=[],  
            textposition='none'  
        ))
        fig3.update_layout(
            width=600,
            height=550,  # Define a altura do gráfico
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            title='',
            title_font_color='orange',
            xaxis_title='',
            yaxis_title='Vendas Realizadas',
            xaxis_title_font_color='orange',
            yaxis_title_font_color='orange',
            xaxis_tickfont_color='orange',
            yaxis_tickfont_color='orange',
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False  
        )        
        # "Top 10 Funcionários que Mais Venderam"
        st.plotly_chart(fig3, use_container_width=True) 

        # "Top 10 Funcionários que Mais Venderam"

    with col20: 
       
        st.subheader("Funcionários com Menos Vendas")

    tail10func = func_rank.tail(5)
   

    with col20:
        fig4 = go.Figure(data=go.Bar(
            x=tail10func.head(10).index, 
            y=tail10func.head(10),
            marker_color='orange',
            text=[],  
            textposition='none'  
        ))
        fig4.update_layout(
           # width=600,
            height=550,  # Define a altura do gráfico
            paper_bgcolor='#0E1117',
            plot_bgcolor='#0E1117',
            title='',
            title_font_color='orange',
            xaxis_title='',
            yaxis_title='Vendas Realizadas',
            xaxis_title_font_color='orange',
            yaxis_title_font_color='orange',
            xaxis_tickfont_color='orange',
            yaxis_tickfont_color='orange',
            margin=dict(l=40, r=20, t=40, b=40),
            showlegend=False  
        )        
        # "Top 10 Funcionários que Mais Venderam"
        st.plotly_chart(fig4, use_container_width=True) 


    st.subheader("Devoluções")
    devolucao = vendas['Devolucoes'].value_counts()
    st.write(devolucao)
        
 
    # st.write("Colunas disponíveis:", vendas.columns)
    #st.subheader("Faturamento Líquido")
    #fatliq = vendas['Vlr_Liquido'].sum()
    #st.write(f'Faturamento líquido: R${fatliq:.2f}')
    #st.subheader("Total de Descontos")
    #st.write(f'Total de descontos: R${total_descontos:.2f}')
    #st.subheader("Total de Produtos Vendidos")
    #totprod = vendas['N_Produtos'].sum()
    #st.write(f'Total de produtos: {totprod:.0f} unidades')
    #st.subheader("Top 10 Funcionários que Mais Venderam")
    #func_rank = vendas['Funcionario'].value_counts()
    #top10func = func_rank.head(10)
    # st.bar_chart(top10func)
    #st.subheader("Funcionários com Menos Vendas")
    #tail10func = func_rank.tail(10)
    #st.bar_chart(tail10func)
    #st.subheader("Devoluções")
    #devolucao = vendas['Devolucoes'].value_counts()
    #st.write(devolucao)
    #st.write("Colunas disponíveis:", vendas.columns)

    
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
        
        # Ajustando o modelo de regressão linear
        model = smf.ols(formula=function, data=vendas).fit()
        
        # Exibindo o resumo do modelo
        st.subheader("Resumo da Regressão Linear")
        st.write(model.summary())
        
        # Preparando os dados para a regressão linear
        x = vendas[['Vlr_Bruto', 'Vlr_Desconto', 'N_Boletos', 'N_Produtos']]
        y = vendas['Vlr_Liquido']
        
        # Separando os dados de treino e teste
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
        
        # Criando o objeto de Regressão Linear
        lr = LinearRegression()
        
        # Treinando o modelo
        lr.fit(x_train, y_train)
        
        # Calculando o coeficiente de determinação (R²)
        r_sq = lr.score(x, y)
        
        col40, col41, col43, col44, col45, col46, col47  = st.columns(7)
        
    
    
    
    
    
        
        # Exibindo o coeficiente de determinação
        #col40.write(f'Coeficiente de Determinação (R²): {r_sq:.2f}')
                     
        # Previsões e métricas de erro para dados de treinamento
        y_pred_train = lr.predict(x_train)
        #col41.write(f'MAE (Treinamento): {metrics.mean_absolute_error(y_train, y_pred_train):.2f}')
        #col43.write(f'MSE (Treinamento): {metrics.mean_squared_error(y_train, y_pred_train):.2f}')
        #col44.write(f'RMSE (Treinamento): {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f}')
        
        # Previsões e métricas de erro para dados de teste
        y_pred = lr.predict(x_test)
        #col45.write(f'MAE (Teste): {metrics.mean_absolute_error(y_test, y_pred):.2f}')
        #col46.write(f'MSE (Teste): {metrics.mean_squared_error(y_test, y_pred):.2f}')
        #col47.write(f'RMSE (Teste): {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f}')
        
        
        col1, col2= st.columns(2)
        #barra_laranja   
        with col1:
                st.markdown(
                f"""
                <style>
                    .regresao {{
                    display: flex;
                    justify-content: center;
                    align-items: center; 
                    gap: 180px;        
                    width: 80.3vw;
                    background-color: #FFA500; 
                    color: black;
                    padding: 10px;
                    border-radius: 10px;
                    text-align: center;
                    font-size: 24px;
                    font-weight: bold;
                    margin-bottom: 50px;
                }}
                .regresao div{{
                 
                 text-align: center;
                 font-size: 16px;
                 font-weight: bold;
                 border-radius: 5px;
                    
                }} 
                </style>
                
                <div class="regresao">
                   <div>  <spam> Coeficiente de Determinação (R²) </spam> <br> {r_sq:.2f} </div>
                   
                   <div> <spam> MAE (Treinamento) </spam> <br> {metrics.mean_absolute_error(y_train, y_pred_train):.2f} </div>
                   
                   <div> <spam> MSE (Treinamento) </spam> <br> {metrics.mean_squared_error(y_train, y_pred_train):.2f} </div>
                   
                   <div> <spam> RMSE (Treinamento) </spam> <br> {np.sqrt(metrics.mean_squared_error(y_train, y_pred_train)):.2f} </div>
                   
                   <div> <spam> MAE (Teste) </spam> <br> {metrics.mean_absolute_error(y_test, y_pred):.2f} </div>
                   
                   <div> <spam> MSE (Teste) </spam> <br> {metrics.mean_squared_error(y_test, y_pred):.2f} </div>
                   
                   <div> <spam> RMSE (Teste) </spam> <br> {np.sqrt(metrics.mean_squared_error(y_test, y_pred)):.2f} </div>
                          
                                         
                   
                </div>
                             
                """,
                unsafe_allow_html=True
            )  
        
        
        
        
        
        # Exibindo o pairplot das variávei
        
        fig, ax = plt.subplots(figsize=(12, 10))  # Ajuste a largura e altura conforme necessário

        # Cria o pairplot
        sns_plot = sns.pairplot(vendas[['Vlr_Bruto', 'Vlr_Desconto', 'Vlr_Liquido', 'N_Produtos']])

        sns_plot.fig.set_size_inches(20, 6)  # Ajuste a largura e altura conforme necessário

        # Define o fundo da figura e dos eixos para preto
        sns_plot.fig.patch.set_facecolor('#0E1117')  # Fundo da figura
        for ax in sns_plot.axes.flatten():
            ax.set_facecolor('#0E1117')  # Fundo dos eixos
            ax.spines['top'].set_color('#f4a460')  # Cor das bordas dos eixos
            ax.spines['right'].set_color('#f4a460')
            ax.spines['left'].set_color('#f4a460')
            ax.spines['bottom'].set_color('#f4a460')
            ax.xaxis.label.set_color('white')  # Cor dos rótulos dos eixos x e y
            ax.yaxis.label.set_color('#f4a460')
            ax.tick_params(axis='both', colors='white')  # Cor dos ticks dos eixos 
            ax.grid(True, color='#f4a460', linestyle='--', linewidth=0.5)  # Cor e estilo da grid

        # Exibe o gráfico no Streamlit
        st.subheader("Pairplot")
        st.pyplot(sns_plot.fig, use_container_width=True)
        
        
        
        
        
        
     

        
        












    else:
        st.error("Dados de vendas não encontrados.")



# Função principal do Streamlit
def main():
    
    
    
    # caminho = st.text_input("Digite o caminho do diretório:", r'D:\PROJETO_REAL_DNC_-SER-PERFORMACE\clientes_EMPRESA-SER')
    caminho =  r'D:\PROJETO_REAL_DNC_-SER-PERFORMACE\clientes_EMPRESA-SER'
    
    if caminho:
        try:
            # Obtendo as pastas dentro do diretório
            pastas = [p for p in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, p))]
            
            if pastas:
                pasta_selecionada = st.sidebar.selectbox("Escolha um cliente:", pastas)
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
