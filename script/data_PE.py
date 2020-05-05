import pandas as pd
import numpy as np


pasta="C:/Users/ravel/Downloads/COVID-19 no Mundo, no Brasil e em Pernambuco (8).csv"
path_out_SUQC_mod = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/data/data_mensured_PE/"

df = pd.read_csv(pasta,header = 0, sep = ",")

df_filtred_confirmed = df.query('classe == "CONFIRMADO"')
df_municipios=pd.pivot_table(df_filtred_confirmed, values = 'X', 
                 columns = ['mun_notificacao'],
                 aggfunc=np.mean)

file_mun = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/data/municipios_PE.csv"
df_municipios.to_csv(file_mun,sep = ";", index = False)
municipios = df_municipios.columns

for mun in municipios:
    
    df_municipio = df_filtred_confirmed.query('mun_notificacao == '+'"'+mun+'"')
    df_municipio["Cases"] = np.zeros(len(df_municipio))+1

    df_organizado = pd.pivot_table(df_municipio, values = 'Cases', 
                                   index = ['dt_notificacao'],
                                   aggfunc=np.sum)

    df_organizado["DateRep"] = pd.to_datetime(df_organizado.index)
    df_corrigido = pd.date_range(start = df_organizado["DateRep"].values[0],
                                 end = df_organizado["DateRep"].values[-1],
                                 freq='D')
    cases = np.zeros(len(df_corrigido))

    for i in range(len(df_organizado)):
        for j in range(len(df_corrigido)):
            if df_corrigido.values[j] == df_organizado["DateRep"].values[i]:
                cases[j] = df_organizado["Cases"].values[i]
            
    df_organizado_ = pd.DataFrame(df_corrigido.values, columns = ["DateRep"])
    df_organizado_["Cases"] = np.array(cases)

    df_organizado_["cum-Cases"] = df_organizado_["Cases"].cumsum()
    df_organizado_["Deaths"] = np.zeros(len(df_organizado_))

    for i in range(len(df_municipio)):
        for j in range(len(df_organizado_)):
            if df_municipio["dt_obito"].values[i] == str(df_corrigido.values[j])[:10]:
                df_organizado_["Deaths"][j]=df_organizado_["Deaths"][j]+1

    df_organizado_["cum-Deaths"] = df_organizado_["Deaths"].cumsum()   
    df_organizado_.to_csv(path_out_SUQC_mod+ "COVID-19 "+ mun + ".csv", sep = ";",index = False)


