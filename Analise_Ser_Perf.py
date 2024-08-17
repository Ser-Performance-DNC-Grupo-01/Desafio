import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
from streamlit_elements import elements, mui, html


st.title("Importação e Processamento de Dados")

# Input para o caminho do diretório
caminho = st.text_input("Digite o caminho do diretório:", r'/home/nickson/DNC-Projetos/Ser_Performance/Clientes')

if caminho:
    try:
        # Lista de pastas no diretório
        pastas = [p for p in os.listdir(caminho) if os.path.isdir(os.path.join(caminho, p))]
        
        # Se houver pastas, exibe um seletor para escolher uma pasta
        if pastas:
            pasta_selecionada = st.selectbox("Escolha uma pasta:", pastas)
            cliente = os.path.join(caminho, pasta_selecionada)
            
            # Dicionário para armazenar os DataFrames
            uploaded_files = {}
            
            # Laço de repetição para ler arquivos CSV e nomear conforme palavras-chave
            for csv in os.listdir(cliente):
                if csv.endswith('.csv'):
                    file_path = os.path.join(cliente, csv)
                    if 'acessos' in csv:
                        uploaded_files['acessos'] = pd.read_csv(file_path)
                    elif 'campanha' in csv:
                        uploaded_files['campanha'] = pd.read_csv(file_path)
                    elif 'feedback' in csv:
                        uploaded_files['feedback'] = pd.read_csv(file_path)
                    elif 'produto' in csv:
                        uploaded_files['produto'] = pd.read_csv(file_path)
                    elif 'treinamento' in csv:
                        uploaded_files['treinamento'] = pd.read_csv(file_path)
                    elif 'vendas' in csv:
                        uploaded_files['vendas'] = pd.read_csv(file_path)
            
            # Verificação e visualização dos dados
            if 'acessos' in uploaded_files:
                acessos = uploaded_files['acessos']
                
                # Exibir as 5 primeiras linhas do DataFrame 'acessos'
                st.subheader("Visualização das 5 primeiras linhas do DataFrame 'acessos'")
                st.write(acessos.head())
                
                # Análise de Acessos por Funcionário
                st.subheader("Acessos por Funcionário")
                if 'Funcionario' in acessos.columns and 'Quantidade_de_Acessos' in acessos.columns:
                    func_acessos = acessos.groupby('Funcionario')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
                    st.write(func_acessos.head(10))
                else:
                    st.warning("O DataFrame 'acessos' não contém as colunas esperadas.")
                
                # Análise de Acessos por Unidade
                st.subheader("Acessos por Unidade")
                if 'Unidade' in acessos.columns and 'Quantidade_de_Acessos' in acessos.columns:
                    unidade = acessos.groupby('Unidade')['Quantidade_de_Acessos'].sum().sort_values(ascending=False)
                    st.write(unidade.head(10))
                else:
                    st.warning("O DataFrame 'acessos' não contém as colunas esperadas.")
            
            else:
                st.warning("O arquivo 'acessos.csv' não foi encontrado na pasta selecionada.")
                
            if 'vendas' in uploaded_files:
                vendas = uploaded_files['vendas']
    
                # Criar a coluna 'Devolucoes'
                vendas['Devolucoes'] = vendas.apply(lambda row: 'S' if pd.isna(row['N_Produtos']) or row['N_Produtos'] <= 0 else 'N', axis=1)
    
                # Exibir as 5 primeiras linhas do DataFrame 'vendas'
                st.subheader("Visualização das 5 primeiras linhas do DataFrame 'vendas'")
                st.write(vendas.head())
    
                # Exibir a distribuição de devoluções
                st.subheader("Distribuição de Devoluções")
                devolucoes_count = vendas['Devolucoes'].value_counts()
                st.write(devolucoes_count)    

                # Calcular o ticket médio
                if vendas.shape[0] > 0 and 'Vlr_Bruto' in vendas.columns:
                    tcktmedio = vendas['Vlr_Bruto'].sum() / vendas.shape[0]
                    st.subheader("Ticket Médio")
                    st.write(f"Ticket médio: R${tcktmedio:.2f}")
                else:
                    st.warning("O DataFrame 'vendas' não contém as colunas esperadas ou está vazio.")
                
                # Calcular e exibir o faturamento bruto
                if 'Vlr_Bruto' in vendas.columns:
                    FatBruto = vendas['Vlr_Bruto'].sum()
                    st.subheader("Faturamento Bruto")
                    st.write(f"Faturamento bruto: R${FatBruto:.2f}")
                else:
                    st.warning("O DataFrame 'vendas' não contém a coluna 'Vlr_Bruto'.")
    
                # Calcular e exibir o faturamento líquido
                if 'Vlr_Liquido' in vendas.columns:
                    fatliq = vendas['Vlr_Liquido'].sum()
                    st.subheader("Faturamento Líquido")
                    st.write(f"Faturamento líquido: R${fatliq:.2f}")
                else:
                    st.warning("O DataFrame 'vendas' não contém a coluna 'Vlr_Liquido'.")
    
                # Calcular e exibir o total de descontos
                if 'Vlr_Desconto' in vendas.columns:
                    total_descontos = vendas['Vlr_Desconto'].sum()
                    st.subheader("Total de Descontos")
                    st.write(f"Total de descontos: R${total_descontos:.2f}")
                else:
                    st.warning("O DataFrame 'vendas' não contém a coluna 'Vlr_Desconto'.")
    
                # Calcular e exibir o total de produtos vendidos
                if 'N_Produtos' in vendas.columns:
                    totprod = vendas['N_Produtos'].sum()
                    st.subheader("Total de Produtos Vendidos")
                    st.write(f"Total de produtos: {totprod:.0f} unidades")
                else:
                    st.warning("O DataFrame 'vendas' não contém a coluna 'N_Produtos'.")
            
            else:
                st.warning("O arquivo 'vendas.csv' não foi encontrado na pasta selecionada.")
    
    except FileNotFoundError:
        st.error("O caminho fornecido não foi encontrado.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")