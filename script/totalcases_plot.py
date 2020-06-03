import pandas as pd
from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cbook import get_sample_data
from matplotlib.ticker import FuncFormatter
from math import log10, floor


def bar_plt(atributo, title_name,df,logscale):
    fig, ax = plt.subplots(1, 1)
    df = df.sort_values(by=[atributo])

    figure = df.plot.bar(ax =ax, x = "Estado", y =atributo,figsize = (15,8), legend = None,width=.75, logy = logscale)
    figure.set_xlabel(" ")
    figure.set_title(title_name, family = "Serif", fontsize = 22)
    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.yaxis.set_major_formatter(plt.FuncFormatter(format_func)) 

    for p in ax.patches:
        b = p.get_bbox()
        val = format_func(b.y1 + b.y0,1)        
        ax.annotate(val, ((b.x0 + b.x1)/2, b.y1 ), fontsize = 14,ha='center', va='top',rotation = 90)

    plt.show()
    path_out ="C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/imagens/"
    fig.savefig(path_out+atributo+'_barplot.png', dpi = 300,bbox_inches='tight',transparent = True)


def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 0)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def desacum(X):
    a = [X[0]]
    for i in range(len(X)-1):
        a.append(X[i+1]-X[i])
    return a

def size_pop(FILE, população):
    for i in população:
        if i[0] == FILE[9:-4]:
            return float(i[1])

file_pop = "C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/populacao.csv"
população = pd.read_csv(file_pop,header=0,sep =";")
população = população.to_numpy()

mypath = 'C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/data_simulated/'
mypath2 = 'C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/data_mensured/'
mypath3 = 'C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/data_simulated_ant/'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
path_out = "C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/imagens/cum_cases/"
path_out_deaths = "C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/imagens/cum_deaths/"


estados = [ "COVID-19 Brazil.CSV", "COVID-19 SC.CSV", "COVID-19 PE.CSV", "COVID-19 SP.CSV", "COVID-19 AM.CSV"]

cont = 0
fig,ax = plt.subplots(1, 1)
Color = np.array(["r","y","g","b","m"])

for i in estados:
      
    estado = i
    pop = size_pop(i,população)
    
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")
    df_mensured = pd.read_csv(mypath2+estado,header = 0 , sep =";")
    
    df_plot = df_simulated[["Cases"]][:-358]
    df_plot["datetime"] = pd.to_datetime(df_simulated["date"])[:-358]
    df_plot['current trend'] = df_simulated["Cases"].values[:-358]
    
    df_plot2 = df_mensured[["cum-Cases"]]
    df_plot2["datetime"] = pd.to_datetime(df_mensured["DateRep"])
    df_plot2[estado[9:-4]] = df_plot2[["cum-Cases"]] 
    
    max_cases = max(df_plot["current trend"])

    figure2 = df_plot2.plot(ax = ax, kind = "line", x = "datetime", y = estado[9:-4],
                             style = Color[cont]+'o-',grid = True,rot = 90,figsize= (10,8), logy = True)
    
    figure = df_plot.plot(ax =ax, kind = "line", x = "datetime", y = 'current trend',
                             color = Color[cont],grid = True,rot = 90,figsize= (10,8), 
                             logy = True, ylim = (1,10**8))
    cont = cont + 1

    figure.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    max_cases = max(df_plot["current trend"])
    max_day = df_plot["datetime"].values[-1:][0]
    val = format_func(max_cases)        
    ax.annotate(val, (max_day, max_cases), fontsize = 14,ha='left', va='bottom')

figure.axvline(pd.to_datetime('2020-03-20'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-03-20'),10**5, "social mitigation", fontsize=12,
               rotation=90, rotation_mode='anchor')

figure.axvline(pd.to_datetime('2020-04-01'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-01'),10**5, "New tests", fontsize=12,
               rotation=90, rotation_mode='anchor')    

figure.axvline(pd.to_datetime('2020-04-17'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-17'),10**5, "change in the \n ministry of health", fontsize=12,
               rotation=90, rotation_mode='anchor')

figure.axvline(pd.to_datetime('2020-04-21'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-21'),10**5, "ICU saturation*", fontsize=12,
               rotation=90, rotation_mode='anchor') 

figure.axvline(pd.to_datetime('2020-04-26'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-26'),10**5, "relaxation of \n the quarantine", fontsize=12,
               rotation=90, rotation_mode='anchor')  

figure.tick_params(axis = 'both', labelsize  = 14)
figure.set_title("Forecast of the total number of cases in 7 days", family = "Serif", fontsize = 18)
figure.set_xlabel(" ")
plt.show()
fig.savefig(path_out + 'allplot'+'.png', dpi = 300,bbox_inches='tight',transparent = True)

cont = 0
inf = []
for i in onlyfiles:
    fig,ax = plt.subplots(1, 1)
        
    estado = i
    pop = size_pop(i,população)
    
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")
    df_mensured = pd.read_csv(mypath2+estado,header = 0 , sep =";")
    
    df_plot = df_simulated[["Cases"]][:-358]
    df_plot["datetime"] = pd.to_datetime(df_simulated["date"])[:-358]
    df_plot[estado[9:-4]] = df_simulated["Cases"].values[:-358]
    
    df_plot2 = df_mensured[["cum-Cases"]]
    df_plot2["datetime"] = pd.to_datetime(df_mensured["DateRep"])
    df_plot2[estado[9:-4]] = df_plot2[["cum-Cases"]]
    
    
    max_cases = max(df_plot[estado[9:-4]])
    max_day = df_plot["datetime"].values[-1:][0]
    
    inf.append([estado[9:-4],max_cases])
    
    
    figure2 = df_plot2.plot(ax = ax, kind = "line", x = "datetime", y = estado[9:-4],
                             style = 'o-',grid = True,rot = 90,figsize= (10,8))
    
    figure = df_plot.plot(ax = ax, x = "datetime", y = estado[9:-4], legend = None,
                             grid = True,rot = 90,figsize= (10,8))
 
  
    figure.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    val = format_func(max_cases)        
    ax.annotate(val, (max_day, max_cases), fontsize = 14,ha='left', va='bottom')

    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.set_title(estado[9:-4], family = "Serif", fontsize = 18)
    figure.set_ylabel("Total Cases", family = "Serif", fontsize = 16)
    figure.set_xlabel(" ")
    plt.close()
    fig.savefig(path_out + estado[:-4]+'.png', dpi = 300,bbox_inches='tight',transparent = True)
    
#    cont = cont+1
#
#    if cont == 3:
#        cont = 0
    
inf_num = []
inf = np.array(inf)
for i in range(len(inf)):
    inf_num.append(inf[i,1].astype(float))
    

df_inf = pd.DataFrame(inf[:,0], columns = ["Estado"])
df_inf["cases_7d"] = np.array(inf_num)
path_out ="C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/inf/"
df_inf.to_csv(path_out+"inf_7d.csv",sep=";")
bar_plt(atributo = "cases_7d", title_name = "Short-term predict (7days)", df = df_inf, logscale = True)

cont = 0
fig,ax = plt.subplots(1, 1)
Color = np.array(["r","y","g","b","m"])

for i in estados:
      
    estado = i
    pop = size_pop(i,população)
    
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")
    df_mensured = pd.read_csv(mypath2+estado,header = 0 , sep =";")
    
    df_plot = df_simulated[["M"]][:-358]
    df_plot["datetime"] = pd.to_datetime(df_simulated["date"])[:-358]
    df_plot['trend'] = df_simulated["M"].values[:-358]
    
    df_plot2 = df_mensured[["cum-Deaths"]]
    df_plot2["datetime"] = pd.to_datetime(df_mensured["DateRep"])
    df_plot2[estado[9:-4]] = df_plot2[["cum-Deaths"]] 
    
    figure2 = df_plot2.plot(ax = ax,kind = "line", x = "datetime", y = estado[9:-4],
                             style = Color[cont]+'o-',grid = True,rot = 90,figsize= (10,8), logy = True)
    
    max_cases = max(df_plot["trend"])
    figure = df_plot.plot(ax = ax,kind = "line", x = "datetime", y = 'trend',
                             color = Color[cont],grid = True,rot = 90,figsize= (10,8), 
                             logy = False, ylim = (1, 10**6))
    cont =cont +1

    figure.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    max_cases = max(df_plot["trend"])
    max_day = df_plot["datetime"].values[-1:][0]
    val = format_func(max_cases)        
    ax.annotate(val, (max_day, max_cases), fontsize = 14,ha='left', va='bottom')

figure.axvline(pd.to_datetime('2020-03-20'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-03-20'),4*10**4, "social mitigation", fontsize=12,
               rotation=90, rotation_mode='anchor')

figure.axvline(pd.to_datetime('2020-04-01'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-01'),4*10**4, "New tests", fontsize=12,
               rotation=90, rotation_mode='anchor')    

figure.axvline(pd.to_datetime('2020-04-17'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-17'),4*10**4, "change in the\nministry of health", fontsize=12,
               rotation=90, rotation_mode='anchor')

figure.axvline(pd.to_datetime('2020-04-21'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-21'),4*10**4, "ICU saturation*", fontsize=12,
               rotation=90, rotation_mode='anchor') 

figure.axvline(pd.to_datetime('2020-04-26'), color='gray', linestyle='--', lw=2)
figure.text(pd.to_datetime('2020-04-26'),4*10**4, "relaxation of \n the quarantine", fontsize=12,
               rotation=90, rotation_mode='anchor')   

figure.tick_params(axis = 'both', labelsize  = 14)
figure.set_title("Forecast of the total Deaths in 7 days", family = "Serif", fontsize = 18)
figure.set_xlabel(" ")
plt.show()
fig.savefig(path_out_deaths + 'all'+'.png', dpi = 300,bbox_inches='tight',transparent = True)


inf = []
for i in onlyfiles:
    fig,ax = plt.subplots(1, 1)
        
    estado = i
    pop = size_pop(i,população)
    
    df_simulated = pd.read_csv(mypath+estado,header = 0 , sep =";")
    df_mensured = pd.read_csv(mypath2+estado,header = 0 , sep =";")
    
    df_plot = df_simulated[["M"]][:-358]
    df_plot["datetime"] = pd.to_datetime(df_simulated["date"])[:-358]
    df_plot[estado[9:-4]] = df_simulated["M"].values[:-358]
    
    df_plot2 = df_mensured[["cum-Deaths"]]
    df_plot2["datetime"] = pd.to_datetime(df_mensured["DateRep"])
    df_plot2[estado[9:-4]] = df_plot2[["cum-Deaths"]]
    
    
    max_cases = max(df_plot[estado[9:-4]])
    max_day = df_plot["datetime"].values[-1:][0]
    
    inf.append([estado[9:-4],max_cases])
    
      
    figure = df_plot.plot(ax =ax, x = "datetime", y = estado[9:-4], legend = None,
                          grid = True,rot = 90,figsize= (10,8))
    
    figure2 = df_plot2.plot(ax = ax,kind = "line", x = "datetime", y = estado[9:-4],
                             style = 'o-',grid = True,rot = 90,figsize= (10,8), logy = True)
      
    figure.yaxis.set_major_formatter(FuncFormatter(format_func))
    
    val = format_func(max_cases)        
    ax.annotate(val, (max_day, max_cases), fontsize = 14,ha='left', va='bottom')

    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.set_title(estado[9:-4], family = "Serif", fontsize = 18)
    figure.set_ylabel("Total Cases", family = "Serif", fontsize = 16)
    figure.set_xlabel(" ")
    plt.close()
    fig.savefig(path_out_deaths + estado[:-4]+'.png', dpi = 300,bbox_inches='tight',transparent = True)
    
#    cont = cont+1
#
#    if cont == 3:
#        cont = 0
    
inf_num = []
inf = np.array(inf)
for i in range(len(inf)):
    inf_num.append(inf[i,1].astype(float))
    

df_inf = pd.DataFrame(inf[:,0], columns = ["Estado"])
df_inf["deaths_7d"] = np.array(inf_num)
path_out ="C:/Users/ravellys/Documents/GitHub/suqcmod_covid19/data/inf/"
df_inf.to_csv(path_out+"inf_deaths_7d.csv",sep=";")
bar_plt(atributo = "deaths_7d", title_name = "Short-term deaths predict (7days)", df = df_inf, logscale = True)
   
