import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.integrate import odeint
from scipy.optimize import curve_fit
import hydroeval as hy
from scipy import stats
from os import listdir
from os.path import isfile, join
from SALib.sample import saltelli
from SALib.analyze import sobol
from math import log10, floor

# Root Mean Square Error
def rrmse(simulation_s, evaluation):
    rrmse_ = np.sqrt(np.mean(((evaluation - simulation_s)/evaluation) ** 2, axis=0, dtype=np.float64))

    return rrmse_

def format_func(value, tick_number=None):
    num_thousands = 0 if abs(value) < 1000 else floor (log10(abs(value))/3)
    value = round(value / 1000**num_thousands, 4)
    return f'{value:g}'+' KMGTPEZY'[num_thousands]

def nstf(ci):
    # Convert to percentile point of the normal distribution.
    pp = (1. + ci) / 2.
    # Convert to number of standard deviations.
    return stats.norm.ppf(pp)

def date_to_str(X):
    str_X=np.datetime_as_string(X, unit='D')
    x=[]
    
    for i in str_X:
        x.append(i[5:7]+'/'+ i[-2:])
    return np.array(X)    

def desacum(X):
    a = [X[0]]
    for i in range(len(X)-1):
        a.append(X[i+1]-X[i])
    return np.array(a)

def R_(t,n1,r1,n2,r2,R1o,R2o):
    R1 = R1o*np.exp(-n1*t)+r1*(1-np.exp(-n1*t))
    R2 = R2o*np.exp(-n2*t)+r2*(1-np.exp(-n2*t))
    return (R1+R2)/2

def R_s(t,n1,r1,n2,r2,R1o,R2o):
#    if (t>1.75/n2):  
#        n2 = n2/2 
    R1 = r1*(np.exp(-n1*t))
    R2 = r2*(1+np.cos(n2*2*np.pi*t))
    return (R1+R2)
 
def sucq(x,t,n1,r1,n2,r2,beta,gama1,gama2,eta,N):
    S=x[0]
    U=x[1]
    Q=x[2]
    C = x[3]
    R1 = x[4]
    R2 = x[5]
    R = x[6]
    M = x[7]
    nC = x[8]
    
    R1t = -n1*(R1-r1)  
    R2t = -n2*(R2-r2)
    Rt = (R1t+R2t)/2
    
    #R = R_s(t,n1,r1,n2,r2,R1,R2)
    alfa = (R*(gama1+gama2)/N)
    St = -alfa*U*S
    Ut = alfa*U*S - gama1*U - gama2*U
    Qt = gama1*U - beta*Q
    Ct = beta*Q
    nCt = gama2*U
    Mt = eta*(Ct)
            
    return [St,Ut,Qt,Ct,R1t,R2t,Rt,Mt,nCt]

def sucq_solve(t,n1,r1,n2,r2,beta,gama1,gama2,eta,N,So,Uo,Qo,Co,R1o,R2o,Ro,nCo):
 
    SUCQ = odeint(sucq, [So,Uo,Qo,Co,R1o,R2o,Ro,0,nCo],t, args=(n1,r1,n2,r2,beta,gama1,gama2,eta,N))
    return SUCQ[:,3].ravel()

def SUCQ(t,n1,r1,n2,r2,beta,gama1,gama2,eta,N,So,Uo,Qo,Co,R1o,R2o,Ro,nCo):
 
    SUCQ = odeint(sucq, [So,Uo,Qo,Co,R1o,R2o,Ro,0,nCo],t, args=(n1,r1,n2,r2,beta,gama1,gama2,eta,N))
    return SUCQ


def ajust_curvefit(days_mens,cumdata_cases,p0,bsup,binf):
    popt, pcov = curve_fit(sucq_solve, days_mens, cumdata_cases,
                           bounds = (binf,bsup),
                           p0 = p0,
                           absolute_sigma = True)
    return popt

from scipy.optimize import minimize

def diferente_zero(X):
    cont = 0
    for i in X:
        if (i<=0):
            cont =cont+1
    return cont        


def object_minimize(x,t,cumdata_cases,cum_deaths):
 
    SUCQ = odeint(sucq, [x[9],x[10],x[11],x[12],x[13],x[14],x[15],0,x[16]],t, args=(x[0],x[1],x[2],x[3],x[4],x[5],x[6],x[7],x[8]))
    w1=1
    w2=1
    r1 = -1*hy.nse(SUCQ[:,3],cumdata_cases)
    r2 = -1*hy.nse(SUCQ[:,7],cum_deaths)

    return (w1*r1+w2*r2)/(w1+w2)

def min_minimize(cum_deaths,cumdata_cases,sucq_solve,p0,t,bsup,binf):
    bnds = ((binf[0],bsup[0]),(binf[1],bsup[1]),(binf[2],bsup[2]),(binf[3],bsup[3]),(binf[4],bsup[4]),(binf[5],bsup[5]),(binf[6],bsup[6]),(binf[7],bsup[7]),(binf[8],bsup[8]),(binf[9],bsup[9]),(binf[10],bsup[10]),(binf[11],bsup[11]),(binf[12],bsup[12]),(binf[13],bsup[13]),(binf[14],bsup[14]),(binf[15],bsup[15]),(binf[16],bsup[16]))
    res = minimize(object_minimize, p0, args = (t,cumdata_cases,cum_deaths), bounds = bnds, method='TNC')
    return res.x

def Ajust_(FILE,pop,extrapolação,day_0,variavel,pasta):    
        
    data_covid = pd.read_csv(pasta+"/"+FILE, header = 0, sep = ";")
    data_covid = data_covid[['DateRep',variavel,"cum-Deaths"]]
    
    nome_data = 'DateRep'
    df = data_covid
    day_0_str=df[nome_data][0][:4]+'-'+df[nome_data][0][5:7]+'-'+df[nome_data][0][-2:]

    #day_0_str = data_covid['DateRep'][0][-4:]+'-'+data_covid['DateRep'][0][3:5]+'-'+data_covid['DateRep'][0][:2]
    date = np.array(day_0_str, dtype=np.datetime64)+ np.arange(len(data_covid))
    
    if date[0]>=np.array(day_0, dtype=np.datetime64):
        t0 = 0
    else:
        dif_dias =np.array(day_0, dtype=np.datetime64)-date[0]
        t0 = dif_dias.astype(int)
    
    date= date[t0:] 
    
    cumdata_covid = data_covid[['Cases']].cumsum()
    cum_deaths = data_covid[['cum-Deaths']]

    cumdata_cases = cumdata_covid['Cases'].values[t0:]
    cum_deaths = cum_deaths['cum-Deaths'].values[t0:]
    t = np.linspace(1,len(cumdata_cases),len(cumdata_cases))
    
    N = pop*10**6
    
    #n1,r1,beta,gama1,N,So,Uo,Qo,Co,Ro
    
    n1_0,r1_0,n2_0,r2_0,beta_0,gama1_0,gama2_0,eta_0=[1/150,14,1/50,-3,.15,.05,.15,2/100] # padrão [1/100,14,1/100,0,.15,.2,2/100]
    So,Uo,Qo,Co,R1o,R2o,nCo = [.8*N,6*cumdata_cases[0],cumdata_cases[0],cumdata_cases[0],-3,14,3*cumdata_cases[0]] # padrão [.9*N,6*cumdata_cases[0],cumdata_cases[0],cumdata_cases[0],0,14]
    p0 = [n1_0,r1_0,n2_0,r2_0,beta_0,gama1_0,gama2_0,eta_0,N,So,Uo,Qo,Co,R1o,R2o,4,nCo] 

    bsup = [n1_0*1.01,18,n2_0*1.1, 0,0.50,.300,0.50, 10/100,N + 1,   N,Uo*2.,Qo*2.0,Co+10**-9, 0,18,6,nCo*2.0]
    binf = [n1_0*0.99,10,n2_0*0.9,-6,0.01,.001,0.05,.01/100,N - 1,.5*N,Uo*.5,Qo*0.5,Co-10**-9,-6,10,4,nCo*0.5]
    
    #p0 = ajust_curvefit(t,cumdata_cases,p0,bsup,binf)
    popt = min_minimize(cum_deaths,cumdata_cases,sucq_solve,p0,t,bsup,binf)
    n1_0,r1_0,n2_0,r2_0,beta_0,gama1_0,gama2_0,eta_0,N,So,Uo,Qo,Co,R1o,R2o,Ro,nCo = popt 

    solution = SUCQ(t,n1_0,r1_0,n2_0,r2_0,beta_0,gama1_0,gama2_0,eta_0,N,So,Uo,Qo,Co,R1o,R2o,Ro,nCo)

    NSE = hy.nse(solution[:,3],cumdata_cases)
    RMSE = hy.rmse(solution[:,3],cumdata_cases)
    MARE = hy.mare(solution[:,3],cumdata_cases)
    
    NSE_deaths = hy.nse(solution[:,7],cum_deaths)
    RMSE_deaths = hy.rmse(solution[:,7],cum_deaths)
    MARE_deaths = hy.mare(solution[:,7],cum_deaths)
    
    print(FILE[9:-4])
    print("n1 = %f " % (n1_0))
    print("n2 = %f " % (n2_0))
    print("r1m = %f" %(r1_0))
    print("r2m = %f" %(r2_0))
    print("R1o = %f" %(R1o))
    print("R2o = %f" %(R2o))
    
    print("beta = %f " % (beta_0))
    print("gamma1 = %f " % (gama1_0))
    print("gamma2 = %f " % (gama2_0))

    print("eta = %f " % (eta_0))
    print("NSE = %.5f " % (NSE))
    print("NSE Deaths = %.5f " % (NSE_deaths))  
    print("#######################")

    date_future = np.array(date[0], dtype=np.datetime64)+ np.arange(len(date)+extrapolação)
    days_future = np.linspace(1,len(cumdata_cases)+extrapolação,len(cumdata_cases)+extrapolação)
    Cum_cases_estimated = SUCQ(days_future, *popt)
    estimativafutura_saída=pd.DataFrame(Cum_cases_estimated[:,3], columns = ["Cases"])
    estimativafutura_saída["S"]=Cum_cases_estimated[:,0]
    estimativafutura_saída["U"]=Cum_cases_estimated[:,1]
    estimativafutura_saída["Q"]=Cum_cases_estimated[:,2]
    estimativafutura_saída["I"] = Cum_cases_estimated[:,1]+Cum_cases_estimated[:,2]+Cum_cases_estimated[:,3]+Cum_cases_estimated[:,8]
    estimativafutura_saída["uC"] =Cum_cases_estimated[:,8]
    estimativafutura_saída["Rt"]=Cum_cases_estimated[:,6]
#    R = []
#    for i in days_future:
#        R.append(R_s(i,n1_0,r1_0,n2_0,r2_0,R1o,R2o))
#    
#    estimativafutura_saída["Rt"]= np.array(R)
    estimativafutura_saída["M"]=Cum_cases_estimated[:,7]
    estimativafutura_saída["date"] = date_future
    path_out = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/data/data_simulated"      
    estimativafutura_saída.to_csv(path_out+'/'+FILE,sep=";")

    return [n1_0,r1_0,n2_0,r2_0,beta_0,gama1_0,gama2_0,eta_0,N,So,Uo,Qo,Co,R1o,R2o,nCo, NSE, RMSE, MARE,NSE_deaths, RMSE_deaths, MARE_deaths]

#import mensured data
população = [["Espanha",46.72],["Itália",60.43],["SP",45.92],["MG",21.17],["RJ",17.26],["BA",14.87],["PR",11.43],["RS",11.37],["PE",9.6],["CE",9.13],["PA",8.6],["SC",7.16],["MA",7.08],["GO",7.02],["AM", 4.14],["ES",4.02],["PB",4.02],["RN",3.51],["MT",3.49],["AL", 3.4],["PI",3.3],["DF",3.1],["MS",2.8],["SE",2.3],["RO",1.78],["TO",1.6],["AC",0.9],["AP",0.85],["RR",0.61],["Brazil",210.2]]
população = np.array(população)

mypath = "C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/data/data_mensured"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

files = [ "COVID-19 SP.CSV"]

extrapolação = 365
day_0 = '2020-02-26'
variavel = 'Cases' 

R = []
estados = [ ]
for i in files:
    FILE = i
    for i in população:
        if i[0] == FILE[9:-4]:
            pop = float(i[1])
    estados.append(FILE[9:-4])
    R.append(Ajust_(FILE,pop,extrapolação,day_0,variavel,pasta = mypath))       
R = np.array(R)
estados = np.array(estados)
df_R = pd.DataFrame(R, columns = ["n1","R1","n2","R2","beta","gamma1","gamma2","eta","N","So","Uo","Qo","Co","R1o","R2o","nCo","NSE","RMSE","MARE","NSE_deaths","RMSE_deaths","MARE_deaths"])
df_R["Estado"] = estados
path_out ="C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/"
df_R.to_csv(path_out+'/metrics.csv',sep=";")

def bar_plt(atributo, title_name,df_R,logscale):
    fig, ax = plt.subplots(1, 1)
    df_R = df_R.sort_values(by=[atributo])

    figure = df_R.plot.bar(ax =ax, x = "Estado", y =atributo,figsize = (15,8), legend = None,width=.75, logy = logscale)
    figure.set_xlabel(" ")
    figure.set_title(title_name, family = "Serif", fontsize = 22)
    figure.tick_params(axis = 'both', labelsize  = 14)
    figure.yaxis.set_major_formatter(plt.FuncFormatter(format_func)) 

    for p in ax.patches:
        b = p.get_bbox()
        val = format_func(b.y1 + b.y0,1)        
        ax.annotate(val, ((b.x0 + b.x1)/2, b.y1), fontsize = 14,ha='center', va='top',rotation = 90)

    plt.show()
    path_out ="C:/Users/ravel/OneDrive/Área de Trabalho/DataScientist/sklearn/COVID-19/CasosPorEstado/suqcmod_covid19/imagens/"
    fig.savefig(path_out+atributo+'_barplot.png', dpi = 300,bbox_inches='tight',transparent = True)

bar_plt(atributo = "NSE", title_name = "NSE", df_R = df_R, logscale = False)
