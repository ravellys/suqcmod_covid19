import pandas as pd
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from math import log10, floor

def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 2)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

mypath = 'C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/data/data_simulated/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
path_out = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/imagens/suqcm/"

for i in onlyfiles:
    fig,ax = plt.subplots(1, 1)
    estado = i
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")
    data = pd.read_csv(mypath+i, header = 0, sep = ";")
    data_covid = data[["Cases","U","Q","S","I"]]
    data_covid["date"] = data["date"]
    data_covid['datetime'] = pd.to_datetime(data_covid['date'])

    
    figure = data_covid.plot(ax = ax, x = "datetime",
                    title = i[9:-4],
                    figsize = (5,4), 
                    grid = True, 
                    rot = 90)#, ylim = (0,pop*10**6))
    figure.legend(loc='center left',bbox_to_anchor=(1.0, 0.5))
    figure.set_ylabel("Individual number", family = "Serif", fontsize = 14)
    figure.set_xlabel("date", family = "Serif", fontsize = 14)
    figure.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    #ax.yaxis.set_major_formatter(plt.FuncFormatter(format_func))
    plt.show()
    fig.savefig(path_out+i[:-4]+".png", dpi = 300,bbox_inches='tight')
    