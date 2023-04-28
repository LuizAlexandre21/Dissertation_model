# Biblioteca 
import numpy as np 
import matplotlib.pyplot as plt 

#Agents
beta  = 0.99
a = 0.756
w = 1.5 
alpha = 0.7
H = 1
h_star = 0.0948 
phi =1
theta  = 0.0918 
tau = 0.275
sigma=0.0

# Função de choque
def z(t):
    if (t >=10) & (t<20) :
        return 1
    else: 
        return 0

# Função de Tecnologia 
def M(K,X_1,X_0):
    # Exponêncial 
    w = 1.5
    expo = X_1*w +alpha*w -w- alpha*X_1 
    m_star = (X_0**(alpha-w))*(alpha**alpha)*((1-tau)**alpha)*((1-alpha)**(w-alpha))*(K**(alpha*(1-w)))
    return m_star**(1/expo)


# Função de impacto do choque de saúde 
def f(K,m):
    return theta*m*K

# Demanda de substencia de saúde 
def h_bar(K,z,m):
    return h_star+z*f(K,m)

# Função de Trabalho 
def L(K,X_1,X_0):
    w=1.5
    expo = X_1*w + alpha*w -w -alpha*X_1 
    l_star = (X_0**(alpha-1))*(alpha**(X_1+alpha-1))*((1-tau)**(X_1+alpha-1))*((1-alpha)**(1-alpha))*(K**((1-alpha)*(X_1-1)))
    return (l_star*(phi**(X_1+alpha-1)))**(1/expo)

# Função de produção 
def Y(K,l,m):
    return (m*K)*(1-alpha)*l**alpha

# Função de Lucro 
def Pi(y,X_1):
    return (1-alpha)*(1-(1/X_1))*y

# Função de salário 
def W(l):
    w=1.5
    return l**(w-1)

# Função de Consumo 
def C(y,m,K,X_1,X_0):
    return y - X_0*((m**X_1)/X_1)*K

# Preços de equilibrio 
def P(K,z,c,l,m):
    w = 1.5
    return ((1-a)/a)*((c-(l**w)/w))/(H-z*f(K,m)-h_star)

# Taxação 
def Tax(W,L):
    return tau*W*L 

# Função de transferencia de recursos para saúde 
def H_pub(Tax,Prices,gamma2):
    return sigma*Tax / (gamma2*Prices)

    
# Consumo dos individuos ricos 
def c_r(K,z,h_pub,pi,p,l,gamma1):
    w=1.5
    return a*(pi + p*H + l*w - pi*h_bar(K,z,m) - (l**w)/w) +(l**w)/w

# Consumo dos individuos pobres
def c_p(K,z,h_pub,pi,p,l):
    w=1.5
    return a*(l**w - p*h_bar(K,z,m) - (l**w)/w +p*h_pub) + (l**w)/w

# Saúde dos individuos ricos 
def h_r(K,z,gamma1,pi,p,l,m):
    w=1.5
    return ((1-a)/p)*(pi + p*(H) + l*w - pi*h_bar(K,z,m) - (l**w)/w) +h_bar(K,z,m)

# Saúde dos individuos pobres
def h_t(K,z,h_pub,pi,p,l,m):
    w=1.5
    return ((1-a)/p)*(l*w - p*h_bar(K,z,m) - (l**w)/w + p*h_pub) + h_bar(K,z,m) 

def h_p(h_t,h_pub):
    return h_t - h_pub

# Testando o codigo 


# Dicionario das regiões 
dic = {'0':{
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
    'omega_t':[]},
    '1':{
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
    'omega_t':[]}}

dic['0']['k'],dic['0']['gamma1'],dic['0']['gamma2'],dic['0']['X_1'],dic['0']['X_0'] = 2,0.22,0.78,5.18,0.22
 
dic['1']['k'],dic['1']['gamma1'],dic['1']['gamma2'],dic['1']['X_1'],dic['1']['X_0'] = 6,0.27,0.73,6.59,0.28


for i in range(100):
    for j in ['0','1']:
        dic[str(j)]['z'].append(z(i))
        K = dic[str(j)]['k']
        X_1 = dic[str(j)]['X_1']
        X_0 = dic[str(j)]['X_0']
        dic[str(j)]['m'].append(M(K,X_1,X_0))
        m = dic[str(j)]['m'][i]
        dic[str(j)]['f'].append(f(K,m))  
        dic[str(j)]['h_bar'].append(h_bar(K,z(i),m))
        dic[str(j)]['l'].append(L(K,X_1,X_0))
        l = dic[str(j)]['l'][i]
        dic[str(j)]['y'].append(Y(K,l,m))
        y = dic[str(j)]['y'][i]
        dic[str(j)]['pi'].append(Pi(y,X_1))
        pi = dic[str(j)]['pi'][i]
        dic[str(j)]['w'].append(W(l))
        w = dic[str(j)]['w'][i]
        dic[str(j)]['c'].append(C(y,m,K,X_1,X_0))
        c = dic[str(j)]['c'][i]
        dic[str(j)]['p'].append(P(K,z(i),c,l,m))
        p = dic[str(j)]['p'][i]
        dic[str(j)]['tax'].append(Tax(w,l))
        tax = dic[str(j)]['tax'][i]
        dic[str(j)]['h_pub'].append(H_pub(tax,p,dic[str(j)]['gamma2']))
        dic[str(j)]['cr'].append(c_r(K,z(i),dic[str(j)]['h_pub'],pi,p,l,dic[str(j)]['gamma1']))
        dic[str(j)]['cp'].append(c_p(K,z(i),dic[str(j)]['h_pub'][i],pi,p,l))
        dic[str(j)]['hr'].append(h_r(K,z(i),dic[str(j)]['gamma1'],pi,p,l,m))
        dic[str(j)]['h_t'].append(h_t(K,z(i),dic[str(j)]['h_pub'][i],pi,p,l,m))
        dic[str(j)]['hp'].append(h_p(dic[str(j)]['h_t'][i],dic[str(j)]['h_pub'][i]))
        dic[str(j)]['omega_c'].append(dic[str(j)]['cr'][i]/dic[str(j)]['cp'][i])
        dic[str(j)]['omega_h'].append(dic[str(j)]['hr'][i]/dic[str(j)]['h_t'][i])
        dic[str(j)]['omega_hnp'].append(dic[str(j)]['hr'][i]/dic[str(j)]['hp'][i])
        dic[str(j)]['omega_t'].append((dic[str(j)]['cr'][i]+dic[str(j)]['hr'][i])/(dic[str(j)]['cp'][i]+dic[str(j)]['hp'][i]))



# Criando Graficos 
fig,ax = plt.subplots(2,3)
# h_bar
ax[0,0].plot(dic['0']['h_bar'])
ax[0,0].plot(dic['1']['h_bar'])
ax[0,0].set_title("H bar")

# Preços 
ax[0,1].plot(dic['0']['p'])
ax[0,1].plot(dic['1']['p'])
ax[0,1].set_title("Preços")

# h_pub 
ax[0,2].plot(dic['0']['h_pub'])
ax[0,2].plot(dic['1']['h_pub'])
ax[0,2].set_title("H pub")

# omega_c
ax[1,0].plot(dic['0']['omega_c'])
ax[1,0].plot(dic['1']['omega_c'])
ax[1,0].set_title('Omega_c')

# omega_h 
ax[1,1].plot(dic['0']['omega_h'])
ax[1,1].plot(dic['1']['omega_h'])
ax[1,1].set_title('Omega_h')

# omega_hnp
ax[1,2].plot(dic['0']['omega_hnp'])
ax[1,2].plot(dic['1']['omega_hnp'])
ax[1,2].set_title('Omega_hnp')
plt.legend(['Rio de Janeiro','São Paulo'])
plt.show()
# omega_t
plt.plot(dic['0']['omega_t'])
plt.plot(dic['1']['omega_t'])
plt.legend(['Rio de Janeiro','São Paulo'])
plt.title('Omega_t')
plt.show()