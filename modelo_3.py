# Biblioteca 
import numpy as np 
import matplotlib.pyplot as plt 

#Agents
beta  = 0.97
a     = 0.756
w = 1.5 
alpha = 0.7
X_0   = 0.22
X_1   = 5.18 
H = 1
h_star = 0.0948 
gamma1 = 0.20
gamma2 = 1-gamma1
h = H / gamma1   
phi =1
theta  = 0.0918 
tau = 0.15
sigma=0.4

# Função de choque
def z(t):
    if (t >=10) & (t<20) :
        return 1
    else: 
        return 0

# Função de Tecnologia 
def m(K):
    # Exponêncial 
    expo = X_1*w +alpha*w -w- alpha*X_1 
    m_star = (X_0**(alpha-w))*(alpha**alpha)*((1-tau)**alpha)*((1-alpha)**(w-alpha))*(K**(alpha*(1-w)))
    return m_star**(1/expo)


# Função de impacto do choque de saúde 
def f(K):
    return theta*m(K)*K

# Demanda de substencia de saúde 
def h_bar(K,z):
    return h_star+z*f(K)

# Função de Trabalho 
def L(K):
    expo = X_1*w + alpha*w -w -alpha*X_1 
    l_star = (X_0**(alpha-1))*(alpha**(X_1+alpha-1))*((1-tau)**(X_1+alpha-1))*((1-alpha)**(1-alpha))*(K**((1-alpha)*(X_1-1)))
    return (l_star*(phi**(X_1+alpha-1)))**(1/expo)

# Função de produção 
def Y(K):
    return (m(K)*K)*(1-alpha)*L(K)**alpha

# Função de Lucro 
def Pi(K):
    return (1-alpha)*(1-(1/X_1))*Y(K)

# Função de salário 
def W(K):
    return (1/phi)*(1/(1-tau))*L(K)**(w-1)

# Função de Consumo 
def C(K):
    return Y(K) - X_0*((m(K)**X_1)/X_1)*K

# Preços de equilibrio 
def P(K,z):
    return ((1-a)/a)*((C(K)-(L(K)**w)/w))/(H-z*f(K)-h_star)

# Taxação 
def Tax(W,L):
    return tau*W*L 

# Função de transferencia de recursos para saúde 
def H_pub(Tax,Prices,gamma2):
    return sigma*Tax / (gamma2*Prices)

    
# Consumo dos individuos ricos 
def c_r(K,z,gamma1):
    return a*(Pi(K)/gamma1 + P(K,z)*(H/gamma1) + L(K)*w - Pi(K)*h_bar(K,z) - (L(K)**w)/w) +(L(K)**w)/w

# Consumo dos individuos pobres
def c_p(K,z,h_pub):
    return a*(L(K)**w - P(K,z)*h_bar(K,z) - (L(K)**w)/w +P(K,z)*h_pub) + (L(K)**w)/w

# Saúde dos individuos ricos 
def h_r(K,z,gamma1):
    return ((1-a)/P(K,z))*(Pi(K)/gamma1 + P(K,z)*(H/gamma1) + L(K)*w - Pi(K)*h_bar(K,z) - (L(K)**w)/w) +h_bar(K,z)

# Saúde dos individuos pobres
def h_t(K,z,h_pub):
     return ((1-a)/P(K,z))*(L(K)*w - P(K,z)*h_bar(K,z) - (L(K)**w)/w + P(K,z)*h_pub) + h_bar(K,z) 

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

dic['0']['k'],dic['0']['gamma1'],dic['0']['gamma2'] = 1,0.22,0.79
 
dic['1']['k'],dic['1']['gamma1'],dic['1']['gamma2'] = 5.44,0.2,0.8


for i in range(100):
    for j in ['0','1']:
        K = dic[str(j)]['k']
        dic[str(j)]['z'].append(z(i))
        dic[str(j)]['m'].append(m(K))
        dic[str(j)]['f'].append(f(K))  
        dic[str(j)]['h_bar'].append(h_bar(K,z(i)))
        dic[str(j)]['l'].append(L(K))
        dic[str(j)]['y'].append(Y(K))
        dic[str(j)]['pi'].append(Pi(K))
        dic[str(j)]['w'].append(W(K))
        dic[str(j)]['c'].append(C(K))
        dic[str(j)]['p'].append(P(K,z(i)))
        dic[str(j)]['tax'].append(Tax(dic[str(j)]['w'][i],dic[str(j)]['l'][i]))
        dic[str(j)]['h_pub'].append(H_pub(dic[str(j)]['tax'][i],dic[str(j)]['p'][i],dic[str(j)]['gamma2']))
        dic[str(j)]['cr'].append(c_r(K,z(i),dic[str(j)]['gamma1']))
        dic[str(j)]['cp'].append(c_p(K,z(i),dic[str(j)]['h_pub'][i]))
        dic[str(j)]['hr'].append(h_r(K,z(i),dic[str(j)]['gamma1']))
        dic[str(j)]['h_t'].append(h_t(K,z(i),dic[str(j)]['h_pub'][i]))
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
plt.legend(['Mendoza (2021)','Brazil (2022)'])
plt.show()
# omega_t
plt.plot(dic['0']['omega_t'])
plt.plot(dic['1']['omega_t'])
plt.legend(['Mendoza (2021)','Brazil (2022)'])
plt.title('Omega_t')
plt.show()