# Biblioteca 
import numpy as np 
import matplotlib.pyplot as plt 

# Parametros Exogenos 
beta = 0.99 
a = 0.756
omega = 1.5
alpha = 0.7 
H = 1
h_star = 0.0948 
phi = 1 
theta = 0.0918
tau = 0.15
sigma = 0.1

# Função Utilidade dos ricos 
def utility(c,l,h,h_bar):
    Agg_first = np.log(c-(l**omega)/omega)
    Agg_second = np.log(h-h_bar)

    return a*Agg_first + (1-a)*Agg_second


# Função de choque 
def z(t):
    if (t>=10) & (t<20):
        return 1 
    else:
        return 0 

# Impacto do choque de saúde 
def f(K,m):
    return theta*m*K

# Nivel de subsistencia de saúde 
def h_bar(z,f):
    return h_star + z*f 

# Taxa de utilização tecnologica 
def m(X_0,X_1,K,tau):
    expo = X_1*omega+alpha*omega-omega-X_1*alpha
    Agg_X0 = X_0**(alpha-omega)
    Agg_tau = (1-tau)**(alpha)
    Agg_alpha = (alpha)**(alpha)
    Agg_alpha1 = (1-alpha)**(omega-alpha)
    Agg_K = K**(alpha*(1-omega))
    return (Agg_X0*Agg_tau*Agg_alpha*Agg_alpha1*Agg_K)**(1/expo )

# Demanda agregada de trabalho 
def l(X_0,X_1,K,tau):
    expo = X_1*omega+alpha*omega-omega-X_1*alpha
    Agg_X0 = X_0**(alpha-1)
    Agg_tau = (1-tau)**(X_1+alpha-1)  
    Agg_alpha= (alpha**(X_1+alpha-1))*(1-alpha)**(1-alpha)
    Agg_K = K**((1-alpha)*(X_1-1))
    return (Agg_X0*Agg_tau*Agg_alpha*Agg_K)**(1/expo)

# Função Produção 
def Y(m,K,L):
    agg_p = (m*K)**(1-alpha)
    agg_L = L**alpha 
    return agg_p * agg_L 

# Função Lucro 
def Pi(y,X_1):
    return (1-alpha)*(1-(1/X_1))*y

# Função Salário 
def W(l,tau):
    return (l**(omega-1))/(1-tau)

# Função consumo agregado 
def C(Y,X_0,m,X_1,K):
    Agg_K = X_0*(m/X_1)*K
    return y-Agg_K

# Preços de equilibrio 
def P(K,z,c,l,m,h_bar,f):
    Agg_a = (1-a)/a
    Agg_l = (l**(omega))/omega
    Agg_num = c - Agg_l
    Agg_den = H-h_star-z*f
    return Agg_a*(Agg_num/Agg_den)

# Taxação
def Tax(w,l,tau):
    return tau*w*l 

# Função de transferencia de recursos para saúdo 
def H_pub(Tax,sigma,Prices,gamma2):
    return (sigma*Tax)/(gamma2*Prices)

# Consumo dos individuos ricos
def c_r(K,z,h_bar,pi,p,gamma1,L,tau):
    h = H/gamma1
    Agg_h = p*h
    Agg_L = (1-tau)*(L)**omega
    Agg_hbar = p*h_bar
    Agg_Lom = (L**omega)/omega
    return a*(pi/gamma1 + Agg_h + Agg_L - Agg_hbar - Agg_Lom) +Agg_Lom


# Consumo dos individuos pobres
def c_p(K,h_bar,h_pub,p,L):
    Agg_L = (L)**omega
    Agg_Lom = (L**omega)/omega 
    Agg_hpub = p*h_pub
    Agg_hbar = p*h_bar 
    return a*(Agg_L-Agg_Lom-Agg_hbar+Agg_hpub)+Agg_Lom

# Consumo de saúde dos individuos ricos
def h_r(K,h_bar,h_pub,pi,p,L,gamma1,tau):
    Agg_a = (1-a)/p 
    h = H/gamma1
    Agg_h = p*h
    Agg_L = (1-tau)*(L)**omega
    Agg_p = p*h_bar 
    Agg_Lom = (L**omega)/omega 
    return Agg_a*(pi/gamma1+Agg_h+Agg_L-Agg_p-Agg_Lom)+h_bar

# Consumo de saúde dos individuos pobres 
def h_t(K,h_bar,h_pub,pi,p,L):
    Agg_a = (1-a)/p 
    Agg_L = (L)**omega 
    Agg_Lom = (L**omega)/omega
    Agg_h = p*h_bar 
    Agg_hpub = p*h_pub 
    return Agg_a*(Agg_L-Agg_Lom-Agg_h+Agg_hpub)+h_bar

# Consumos unico dos individuos privados 
def h_p(h_t,h_pub):
    return h_t - h_pub

# Calculando a taxa de juros das cidade
def interest(c:list,L:list):
    Agg_num = c[0] - ((L[0])**omega)/omega 
    Agg_den = beta*(c[1] - ((L[1])**omega)/omega )
    return Agg_num/Agg_den


# Criando as cidades 
def cidades(n):
    dic = {}
    for i in range(n):
        dic[str(i)] ={
        'z':[],
        'm':[],
        'f':[],
        'l':[],
        'h_bar':[],
        'h_pub':[],
        'h_t':[],
        'y':[],
        'pi':[],
        'w':[],
        'c':[],
        'p':[],
        'cr':[],
        'cp':[],
        'hr':[],
        'hp':[],
        'tax':[],
        'omega_c':[],
        'omega_h':[],
        'omega_hnp':[],
        'omega_t':[],
        'omega_country':[],
        'ur':[],
        'up':[],}

    return dic


# Construindo o modelo 
# Definindo as regiões 
cidade = cidades(3)

# Definindo os parametros locais 
# Região 0 - Rio de Janeiro 
# Capital 
cidade['0']['K'] = 1

# Proporção de ricos 
cidade['0']['gamma1'] = 0.22

# Proporção de pobres 
cidade['0']['gamma2'] = 0.78

# Chi_0 
cidade['0']['X_0'] = 0.22

# Chi_1 
cidade['0']['X_1'] = 5.18

# Sigma
cidade['0']['sigma'] = sigma
#cidade['0']['sigma'] = sigma*1.4

# Tax 
cidade['0']['tau'] = tau
#cidade['0']['tau'] = tau*0.6


# Região 1 - São Paulo 
# Capital 
cidade['1']['K'] = 3

# Proporção de ricos 
cidade['1']['gamma1'] = 0.27 

# Proporção de pobres 
cidade['1']['gamma2'] = 0.73 

# Chi_0 
cidade['1']['X_0'] = 0.28

# Chi_1 
cidade['1']['X_1'] = 6.59

# Sigma
cidade['1']['sigma'] = sigma
#cidade['1']['sigma'] = sigma*0.6

# Tax 
#cidade['1']['tau'] = tau
cidade['1']['tau'] = tau*1.4

# Região 2 - Mendoza 

cidade['2']['K'] = 6.04

# Proporção de ricos 
cidade['2']['gamma1'] = 0.2

# Proporção de pobres 
cidade['2']['gamma2'] = 0.8

# Chi_0 
cidade['2']['X_0'] = 0.1

# Chi_1 
cidade['2']['X_1'] = 6.1

# Construindo a dinamica do modelo 
# Periodos de tempo 
T = 100 

# Interações 
for t in range(T):
    # Interando as cidades
    for town in ['0','1']:

        # Calculando os choques de saúde 
        Z = z(t)
        cidade[town]['z'].append(Z)

        # Calculando o choque de utilização tecnologica 
        M= m(K=cidade[town]['K'],X_1 = cidade[town]['X_1'], X_0 = cidade[town]['X_0'],tau = cidade[town]['tau'])
        cidade[town]['m'].append(M) 

        # Calculando o choque de trabalho 
        L = l(K=cidade[town]['K'],X_1= cidade[town]['X_1'],X_0=cidade[town]['X_0'],tau = cidade[town]['tau'])
        cidade[town]['l'].append(L)

        # Calculando a função f 
        F = f(K=cidade[town]['K'],m=M)
        cidade[town]['f'].append(F)

        # Calculando o h_bar 
        HB = h_bar(f=F,z=Z)
        cidade[town]['h_bar'].append(HB) 
        
        # Calculando o produto 
        y = Y(m=M,K=cidade[town]['K'],L=L)
        cidade[town]['y'].append(y)

        # Calculando o Lucro 
        pi = Pi(y=y,X_1=cidade[town]['X_1'])
        cidade[town]['pi'].append(pi)

        # Calculando o salário 
        w = W(L,tau=cidade[town]['tau'])
        cidade[town]['w'].append(w)

        # Calculando o Consumo 
        c = C(Y=y,X_0=cidade[town]['X_0'],m=M,X_1=cidade[town]['X_1'],K=cidade[town]['K'])
        cidade[town]['c'].append(c)

        # Calculando os Preços 
        p = P(K=cidade[town]['K'],z=Z,c=c,l=L,m=M,h_bar=HB,f=F)
        cidade[town]['p'].append(p)

        # Taxação 
        tax = Tax(w=w,l=L,tau=cidade[town]['tau'])
        cidade[town]['tax'].append(tax)

        # Saúde Publica 
        Hpub = H_pub(Tax=tax,sigma=cidade[town]['sigma'],Prices=p,gamma2=cidade[town]['gamma2']) 
        cidade[town]['h_pub'].append(Hpub)

        # Consumo de bens não saúde dos agentes ricos 
        cr = c_r(K=cidade[town]['K'],z=Z,h_bar=HB,pi=pi,p=p,gamma1=cidade[town]['gamma1'],L=L,tau=cidade[town]['tau'])
        cidade[town]['cr'].append(cr)

        # Consumo de bens de saúde dos agentes ricos 
        hr = h_r(K=cidade[town]['K'],h_bar=HB,h_pub=Hpub,pi=pi,p=p,L=L,gamma1=cidade[town]['gamma1'],tau=cidade[town]['tau'])
        cidade[town]['hr'].append(hr)
    
        # Consumo de bens não saúde dos agentes pobres
        cp = c_p(K=cidade[town]['K'],h_bar=HB,h_pub=Hpub,p=p,L=L)
        cidade[town]['cp'].append(cp)

        # Consumo de bens saúde dos agentes pobres
        ht = h_t(K=cidade[town]['K'],h_bar=HB,h_pub=Hpub,pi=pi,p=p,L=L)
        cidade[town]['h_t'].append(ht)

        # Consumo de bens de saúde privado dos agentes pobres
        hp = h_p(h_t=ht,h_pub=Hpub)
        cidade[town]['hp'].append(hp)

        # Calculando a utilidade indireta dos ricos
        ur = utility(c=cr,l=L,h=hr,h_bar=HB)
        cidade[town]['ur'].append(ur)
        # Calculando a utilidade indireta dos pobres
        up = utility(c=cp,l=L,h=hp,h_bar=HB)
        cidade[town]['up'].append(up)

        # Calculando a desigualdade de consumo 
        omegac = cr/cp
        cidade[town]['omega_c'].append(omegac)

        # Calculando a desigualdade de consumo de bens de saúde 
        omegah = hr/ht
        cidade[town]['omega_h'].append(omegah)

        # Calculando a desigualdade de consumo publico 
        omegahnp = hr/hp 
        cidade[town]['omega_hnp'].append(omegahnp)

        # Calculando a desigualdade total 
        omegat = (cr+hr)/(cp+ht)
        cidade[town]['omega_t'].append(omegat)


omega_con = (cidade['0']['cr'] +cidade['1']['cr'] ) / (cidade['0']['cp'] +cidade['1']['cp'] ) 

# Criando Graficos 
fig,ax = plt.subplots(2,3)
# h_bar
ax[0,0].plot(cidade['0']['h_bar'])
ax[0,0].plot(cidade['1']['h_bar'])
ax[0,0].set_title("H bar")

# Preços 
ax[0,1].plot(cidade['0']['p'])
ax[0,1].plot(cidade['1']['p'])
ax[0,1].set_title("Preços")

# h_pub 
ax[0,2].plot(cidade['0']['h_pub'])
ax[0,2].plot(cidade['1']['h_pub'])
ax[0,2].set_title("H pub")

# omega_c
ax[1,0].plot(cidade['0']['omega_c'])
ax[1,0].plot(cidade['1']['omega_c'])
ax[1,0].set_title('Omega_c')

# omega_h 
ax[1,1].plot(cidade['0']['omega_h'])
ax[1,1].plot(cidade['1']['omega_h'])
ax[1,1].set_title('Omega_h')

# omega_hnp
ax[1,2].plot(cidade['0']['omega_hnp'])
ax[1,2].plot(cidade['1']['omega_hnp'])
ax[1,2].set_title('Omega_hnp')
plt.legend(['Rio de Janeiro','São Paulo'])
plt.show()
# omega_t
plt.plot(cidade['0']['omega_t'])
plt.plot(cidade['1']['omega_t'])
plt.legend(['Rio de Janeiro','São Paulo'])
plt.title('Omega_t')
plt.show()
