from os import sep
from git import Git
import pandas as pd
import numpy as np 
import streamlit as st
from pycaret.classification import load_model, predict_model
from pycaret import *
import PIL
from PIL import Image
from sklearn.metrics import *
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.utils import check_metric
from streamlit_lottie import st_lottie
from streamlit_echarts import st_echarts
import requests
# remove warnings do streamlit
st.set_option('deprecation.showPyplotGlobalUse', False)


def load_lottieurl(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()
ico = Image.open('ico.ico')
st.set_page_config(
    page_title="SVM Team",
    page_icon= ico,    
    layout="wide", #centered",
    initial_sidebar_state='auto',
    menu_items=None)
paginas = ['Home','Análise python','Análise de Churn BI', 'Data Science', "Demonstação","Filtro", "Predição de Churn","Consulta Cliente", "Dashbord comparativo"]
site = ""
site_pred = ""
###### SIDE BAR ######
col1, col2, col3 = st.sidebar.columns([0.5, 1, 1])
with col2:
    image1 = Image.open('logo_size.jpg')
    st.image(image1, width=120)
    pagina = st.sidebar.selectbox("Navegação", paginas)
###### PAGINA INICIAL ######
if pagina == 'Home':
    lottie_1 = load_lottieurl('https://assets2.lottiefiles.com/packages/lf20_3FIGvm.json')
    st_lottie(lottie_1, speed=0.3, height=150, key="initial")    
    col1,col2,col3 = st.columns([1,2,3])
    site = "https://docs.google.com/presentation/d/e/2PACX-1vRuiyA_eVhRcdNymAubZKRNTewo3f2zpg1KZbqrMu2nBhkh7C_XBeBHyp74Efost0X0jsMKCxLULA1_/embed?start=false&loop=false&delayms=3000"
    st.components.v1.iframe(site, width=960, height=569)     
###### PAGINA Análise python ######
if pagina == 'Análise python':
    lottie_2 = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_m9zragkd.json')
    st_lottie(lottie_2, speed=0.5, height=150, key="initial")
    st.subheader("Problema do Negócio")
    HtmlFile = open("Analise_churn.html", 'r', encoding='utf-8')
    source_code = HtmlFile.read() 
    #print(source_code)
    st.components.v1.html(source_code,height = 27500)  


###### BI ######
if pagina == 'Análise de Churn BI':
    st.subheader("Análise de Churn")    
    col1,col2,col3 = st.columns([1,2,3])
    site = "https://bi"
    st.components.v1.iframe(site, width=960, height=600, scrolling=True)

    st.sidebar.write("""O Dashbord: exemplificar.   
    
    """)
###### PAGINA Data Science ######
if pagina == 'Data Science':
    lottie_3 = load_lottieurl('https://assets9.lottiefiles.com/packages/lf20_q5qeoo3q.json')
    st_lottie(lottie_3, speed=0.5, height=150, key="initial")
    st.subheader("Proposta de solução usando Machine Learning")
    HtmlFile = open("DataScience.html", 'r', encoding='utf-8')
    ds_code = HtmlFile.read() 
    #print(source_code)
    st.components.v1.html(ds_code,height = 34000)    

###### Demonstação do modelo de machine learning ######
if pagina == 'Demonstação':
    st.sidebar.write("""Nesse exemplo: o cliente irá carregar a outra parte da base de dados, essa base deve estar no mesmo formato da primeira base.
    obs.: caso desejado poderá utilizar um arquivo de exemplo 'validação_base.csv' basta copiar esse link: [validação_base.csv](https://raw.githubusercontent.com/Jcnok/Stack_Labs_Churn/main/Data/valida%C3%A7%C3%A3o_base.csv) ao clicar em Browse files cole o caminho e clique em abrir.
     
    """)

    st.markdown("### Carregue a base de dados no formato .csv contendo o restante da base de dados dos Clientes")
    st.markdown("---")
    uploaded_file = st.file_uploader("escolha o arquivo *.csv")
    if uploaded_file is not None:
        dados = pd.read_csv(uploaded_file)
        #dados = pd.read_csv(uploaded_file)
        st.write(dados.head()) # checar a saída no terminal
    
    if st.button('CLIQUE AQUI PARA EXECUTAR O MODELO'):
        modelo = load_model('./lgbm_tune_pycaret') 
        pred = predict_model(modelo, data = dados)        
        classe_churn = pred.query('Exited == 1')
        classe_label_churn = pred.query('Label == 1')['Label'].count()
        count_pred = (pred['Label'] == 1).sum()
        count_pred_Label = (classe_churn['Label']==1).sum()
        count_total = (classe_churn['Exited']).count()
        recal = check_metric(pred['Exited'], pred['Label'], metric='Recall')
        result = f'''
        Em um total de {classe_label_churn} chutes, o modelo foi capaz de identificar {count_pred_Label} clientes, dos {count_total} que realmente saíram por algum motivo. Dentro da lista, o modelo encontrou {round((recal * 100),2)}% de todos os clientes que deram churn. Lembrando que apesar de um bom resultado, essa é apenas uma demonstração. Podemos facilmente melhorar essa precisão.'''
        st.subheader(result)
        st.markdown("---")
        st.markdown('### Caso desejado, você pode realizar o download do resultado no formato .csv clicando logo abaixo!')
        @st.cache
        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun        
            return df.to_csv().encode('utf-8')               
        csv = convert_df(pred)        
        st.download_button(
        label="Download do aquivo .CSV",
        data=csv,
        file_name='predict.csv',
        mime='text/csv',
        )
####### Filtro para o modelo #############
if pagina == 'Filtro':
    dados = pd.read_csv('https://raw.githubusercontent.com/Jcnok/Stack_Labs_Churn/main/Data/valida%C3%A7%C3%A3o_base.csv')
    modelo = load_model('./lgbm_tune_pycaret') 
    pred = predict_model(modelo, data = dados)    
    lista_score = (pred.Score.unique()).round(2)     
    lista_score = sorted(lista_score)
    minimo, maximo = st.sidebar.select_slider('Selecione o filtro desejado:',
    lista_score,value=[min(lista_score),max(lista_score)])
    lista = (pred.query(f'Score <= {maximo} and Score >= {minimo}'))
    st.write(lista.tail())
    
    # Função para o conjunto de validação.
    def test_score_report(data_unseen, predict_unseen):
        accuracy = accuracy_score(data_unseen["Exited"], predict_unseen["Label"])
        roc_auc = roc_auc_score(data_unseen["Exited"], predict_unseen["Label"])
        precision = precision_score(data_unseen["Exited"], predict_unseen["Label"])
        recall = recall_score(data_unseen["Exited"], predict_unseen["Label"])
        f1 = f1_score(data_unseen["Exited"], predict_unseen["Label"])
        shape = data_unseen.shape[0]

        df_unseen = pd.DataFrame({
            "Acurácia" : [accuracy],
            "AUC" : [roc_auc],
            "Recall" : [recall],
            "Precisão" : [precision],
            "F1 Score" : [f1],
            "Tamanho do Conjunto":[shape]
        })
        return df_unseen
    # Confusion Matrix
    def conf_mat(data_unseen, predict_unseen):
        unique_label = data_unseen["Exited"].unique()
        cmtx = pd.DataFrame(
            confusion_matrix(data_unseen["Exited"],
                            predict_unseen["Label"],
                            labels=unique_label), 
            index=['{:}'.format(x) for x in unique_label], 
            columns=['{:}'.format(x) for x in unique_label]
        )
        ax = sns.set(rc={'figure.figsize':(4,2)})
        ax = sns.heatmap(cmtx, annot=True, fmt="d", cmap="YlGnBu")
        ax.set_ylabel('Predito')
        ax.set_xlabel('Real')
        ax.set_title("Matriz de Confusão do conjunto de Validação", size=10)
        return st.pyplot()       
    st.write(test_score_report(lista, lista))    
    conf_mat(lista, lista)
    @st.cache
    def convert_df(df):             
        return df.to_csv(sep=';',decimal='.',index=False).encode('utf-8')

    list_class_1 = lista.query('Label == 1')
    st.markdown('### Análise estatística dos Clientes classificados como Churn.')
    st.write(list_class_1[['CreditScore','Age','Tenure','NumOfProducts','Balance','EstimatedSalary']].describe())                   
    st.markdown('### Soma do Saldo e do Salarário dos clientes classificados como Churn.')
    st.write(list_class_1[['Balance','EstimatedSalary']].sum())
    st.markdown('### Opção para salvar o filtro da lista dos clientes!')
    csv = convert_df(list_class_1)        
    st.download_button(
    label="Download do aquivo .CSV",
    data=csv,
    file_name='churn_high.csv',
    mime='text/csv',
    )
                   
    
###### Modelo de predição ######
if pagina == 'Predição de Churn':
    st.markdown('### Selecione as opções de acordo com os dados dos Clientes e execute o modelo!')
    st.sidebar.write("""Aqui o cliente consegue selecionar os dados do perfil do cliente de forma individual.
     O modelo irá informar se esse perfil tem ou não uma tendência maior ao Churn.
    """)
        
    st.markdown('---')    
    sexo = st.radio('Selecione o Sexo',['MASCULINO', 'FEMININO'])
    idade = np.int64(st.slider('Entre com a idade:', 18, 92, 38))	
    pais = st.selectbox('Informe o País:',['França', 'Alemanha', 'Espanha'])
    qtd_produtos = st.selectbox('Quantidade de produtos:',[1,2,3,4])
    tenure = st.selectbox('Tempo de permanência:', [0,1,2,3,4,5,6,7,8,9,10])
    tem_cartao = st.radio('Possui cartão de Crédito:', ['Sim','Não'])
    membro_ativo = st.radio('É membro ativo:', ['Sim','Não'])
    score = np.int64(st.slider('Crédito Score:', 350,850,650))
    salario = np.int64(st.slider('Selecione o Salário estimado:', 10,200000,100000)) 
    saldo = np.float64(st.slider('Selecione o Saldo em conta:',0,251000, 76500))
   
    st.markdown('---')
        
    dic = {'Gender': [sexo], 'Age': [idade], 'Geography': [pais],'NumOfProducts': [qtd_produtos],
            'Tenure': [tenure], 'HasCrCard': [tem_cartao],'IsActiveMember':[membro_ativo],
            'CreditScore':[score],'Balance':[saldo],'EstimatedSalary': [salario]}  
    teste = pd.DataFrame(dic)

    teste["HasCrCard"] = teste["HasCrCard"].map({ 0 : 'Não', 1 : 'Sim'})
    teste["IsActiveMember"] = teste["IsActiveMember"].map({ 0 : 'Não', 1 : 'Sim'})
    teste["Gender"] = teste["Gender"].map({ 'Male' : 'MASCULINO', 'Female' : 'FEMININO'})     


    if st.button('CLIQUE AQUI PARA EXECUTAR O MODELO'):
        modelo = load_model('./lgbm_tune_pycaret') 
        pred_test = predict_model(modelo, data = teste)
        #prob = list(pred_test.Score.round(2)*100)
        value = ((pred_test.Score.astype('float')[0])*100).round(2)
        
        if pred_test.Label.values == 1:
            ##função Js para Grafico de Gauge.
            color = [[0.25, '#ffa173'],[0.5, '#fa6644'],[0.75, '#f52c15'],[1, '#900000']]
            option = {
                    "series": [
                        {
                            "type": 'gauge',
                            "startAngle": 180,
                            "endAngle": 0,
                            "min": 50,
                            "max": 100,
                            "splitNumber": 8,
                            "axisLine": {
                                "lineStyle": {
                                "width": 6,
                                "color": color                            
                            }
                            },
                        "pointer": {
                                "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                                "length": "12%",
                                "width": 20,
                                "offsetCenter": [0, '-60%'],
                                "itemStyle": {
                                "color": 'auto'
                                }
                            },
                        "axisTick": {
                                "length": 12,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 2
                                }
                            },
                        "splitLine": {
                                "length": 20,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 5
                                }
                            },
                        "axisLabel": {
                                "color": '#464646',
                                "fontSize": 15,
                                "distance": -60
                                                        
                            },
                        "title": {
                                "offsetCenter": [0, '-20%'],
                                "fontSize": 20
                            },
                        "detail": {
                                "fontSize": 20,
                                "offsetCenter": [0, '0%'],
                                "valueAnimation": "true",
                                "formatter":value,                           
                                "color": 'auto'
                            },
                        "data": [
                                {
                                "value": value,
                                "name": 'Churn Rating'
                                }
                            ]
                        }
                    ]
                }; 
            st.markdown(f'### Probabilidade do cliente Cancelar o serviço: {value}%.')            
            ##Plot Gauge
            st_echarts(options=option, width="100%", key=value)             
            
        else:
            ##função Js para Grafico de Gauge.
            color = [[0.25, '#7eab70'],[0.5, '#659259'],[0.75, '#4d7841'],[1, '#056003']]
            option = {
                    "series": [
                        {
                            "type": 'gauge',
                            "startAngle": 180,
                            "endAngle": 0,
                            "min": 50,
                            "max": 100,
                            "splitNumber": 8,
                            "axisLine": {
                                "lineStyle": {
                                "width": 6,
                                "color": color                            
                            }
                            },
                        "pointer": {
                                "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                                "length": "12%",
                                "width": 20,
                                "offsetCenter": [0, '-60%'],
                                "itemStyle": {
                                "color": 'auto'
                                }
                            },
                        "axisTick": {
                                "length": 12,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 2
                                }
                            },
                        "splitLine": {
                                "length": 20,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 5
                                }
                            },
                        "axisLabel": {
                                "color": '#464646',
                                "fontSize": 15,
                                "distance": -60
                                                        
                            },
                        "title": {
                                "offsetCenter": [0, '-20%'],
                                "fontSize": 20
                            },
                        "detail": {
                                "fontSize": 20,
                                "offsetCenter": [0, '0%'],
                                "valueAnimation": "true",
                                "formatter":value,                           
                                "color": 'auto'
                            },
                        "data": [
                                {
                                "value": value,
                                "name": 'Churn Rating'
                                }
                            ]
                        }
                    ]
                }
            st.markdown(f'### Probabilidade do cliente permanecer com o serviço: {value}%.')    
            ##Plot Gauge
            st_echarts(options=option, width="100%", key=value)

###### Consulta Cliente ########
if pagina == "Consulta Cliente" :
    df = pd.read_csv("https://raw.githubusercontent.com/Jcnok/Stack_Labs_Churn/main/Data/Churn_Modelling.csv")
    ids = df.CustomerId.unique() 
    modelo = load_model('./lgbm_tune_pycaret')
    id = st.number_input("Informe o ID do Cliente",ids.min(),ids.max())
    if id in ids:
        filtro = df.query(f'CustomerId=={id}')
        st.dataframe(filtro)
        pred_filtro = predict_model(modelo,data=filtro)
        value = round((pred_filtro.Score.astype('float').to_list()[0])*100,2)
        if pred_filtro.Label.values == 1:
            ##função Js para Grafico de Gauge.
            color = [[0.25, '#ffa173'],[0.5, '#fa6644'],[0.75, '#f52c15'],[1, '#900000']]
            option = {
                    "series": [
                        {
                            "type": 'gauge',
                            "startAngle": 180,
                            "endAngle": 0,
                            "min": 50,
                            "max": 100,
                            "splitNumber": 8,
                            "axisLine": {
                                "lineStyle": {
                                "width": 6,
                                "color": color                            
                            }
                            },
                        "pointer": {
                                "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                                "length": "12%",
                                "width": 20,
                                "offsetCenter": [0, '-60%'],
                                "itemStyle": {
                                "color": 'auto'
                                }
                            },
                        "axisTick": {
                                "length": 12,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 2
                                }
                            },
                        "splitLine": {
                                "length": 20,
                                "lineStyle": {
                                "color": 'auto',
                                "width": 5
                                }
                            },
                        "axisLabel": {
                                "color": '#464646',
                                "fontSize": 15,
                                "distance": -60
                                                        
                            },
                        "title": {
                                "offsetCenter": [0, '-20%'],
                                "fontSize": 20
                            },
                        "detail": {
                                "fontSize": 20,
                                "offsetCenter": [0, '0%'],
                                "valueAnimation": "true",
                                "formatter":value,                           
                                "color": 'auto'
                            },
                        "data": [
                                {
                                "value": value,
                                "name": 'Churn Rating'
                                }
                            ]
                        }
                    ]
                }; 
            st.markdown(f'### Probabilidade do cliente Cancelar o serviço: {value}%.')            
            ##Plot Gauge
            st_echarts(options=option, width="100%", key=value)             
        else:
                ##função Js para Grafico de Gauge.
                color = [[0.25, '#7eab70'],[0.5, '#659259'],[0.75, '#4d7841'],[1, '#056003']]
                option = {
                        "series": [
                            {
                                "type": 'gauge',
                                "startAngle": 180,
                                "endAngle": 0,
                                "min": 50,
                                "max": 100,
                                "splitNumber": 8,
                                "axisLine": {
                                    "lineStyle": {
                                    "width": 6,
                                    "color": color                            
                                }
                                },
                            "pointer": {
                                    "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                                    "length": "12%",
                                    "width": 20,
                                    "offsetCenter": [0, '-60%'],
                                    "itemStyle": {
                                    "color": 'auto'
                                    }
                                },
                            "axisTick": {
                                    "length": 12,
                                    "lineStyle": {
                                    "color": 'auto',
                                    "width": 2
                                    }
                                },
                            "splitLine": {
                                    "length": 20,
                                    "lineStyle": {
                                    "color": 'auto',
                                    "width": 5
                                    }
                                },
                            "axisLabel": {
                                    "color": '#464646',
                                    "fontSize": 15,
                                    "distance": -60
                                                            
                                },
                            "title": {
                                    "offsetCenter": [0, '-20%'],
                                    "fontSize": 20
                                },
                            "detail": {
                                    "fontSize": 20,
                                    "offsetCenter": [0, '0%'],
                                    "valueAnimation": "true",
                                    "formatter":value,                           
                                    "color": 'auto'
                                },
                            "data": [
                                    {
                                    "value": value,
                                    "name": 'Churn Rating'
                                    }
                                ]
                            }
                        ]
                    }
                st.markdown(f'### Probabilidade do cliente permanecer com o serviço: {value}%.')    
                ##Plot Gauge
                st_echarts(options=option, width="100%", key=value)
    else:
        st.markdown("### Cliente inexistente, informe um id válido!")    

###### Dashboard Compartivo ######
if pagina == 'Dashbord comparativo':    
    st.subheader("Dashboard compartivo entre o resultado real Vs resultado do modelo")    
    col1,col2,col3 = st.columns([1,2,3])
    st.components.v1.iframe(site_pred, width=1400, height=800, scrolling=True)

    st.sidebar.write("""O Dashbord é interativo, posicione o mouse sobre os gráficos para obter o comparativo de acertos.    
    """)
        
        

        