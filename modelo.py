# Biblioteca 
import numpy as np 

# Parametros 
np.random.seed(3424) 
alpha = 0.7
beta = 0.99 
w = 1
m = 1
X_0 = 1.0
X_1 = 6.1
gamma = 0.2 
H =1
h_star = 0.0948 
h=H/gamma
a = 0.756 
theta = 0.0918
tau = 0.0
T = 100
sigma = 0.1 

# Função phi
def phi(z):
    return 1 - 0.2*z

# Função X 
def X(poor:list,rich:list,K):
    Phi = sum([(len(rich)/(len(rich)+len(poor)))*elem for elem in rich]) + sum([(len(poor)/(len(rich)+len(poor)))*elem for elem in poor])
    phi = rich[0]
    return ((Phi/phi)**(w-1))*(1-tau)*phi

# Função progresso tecnologico
def m(X,K):
    m_star =(X_0**(alpha-w))*(alpha**alpha)*(1-alpha)**(w-alpha)*(K**(alpha*(1-w)))
    expo = 1/(X_1*w+alpha*w-w-X_1*alpha)
    phi_t =  X
    return ((X**alpha)*m_star)**expo

# Função demanda de bens de saúde 
def h_bar(m,z,K):
    f_t = theta*m*K 
    return h_star + z*f_t

# Função saúde das classes sociais 
def h_social(prices,L,h_pub,h_bar,social,Pi,gamma1):
    if social =='Poors':
        return ((1-a)/prices)*(0 + L**w - prices*h_bar - (L**w)/w + prices*h_pub) + h_bar - h_pub
    elif social == 'Richs':
        return ((1-a)/prices)*(Pi/gamma1 + prices*(H/gamma1) + L**w - prices*h_bar - (L**w)/w) + h_bar

# Função consumo de classes sociais 
def c_social(prices,L,h_pub,h_bar,social,Pi,gamma1,):
    if social == 'Poors':
        return a*(L**w - prices*h_bar - (L**w)/w + prices*h_pub) + (L**w)/w
    elif social == 'Richs':
        return a*(Pi/gamma1 + prices*(H/gamma1) + L**w - prices*h_bar - (L**w)/w) + (L**w)/w

# Função de preços 
def prices(C,L,h_bar):
     return ((1-a)/a)*((C - ((L**w)/w)))/(H - h_bar)

# Função trabalho 
def L(X,K):
    L_star = (X_0**(alpha-1))*(alpha**(X_1+alpha-1))*((1-tau)**(X_1+alpha-1))*((1-alpha)**(1-alpha))*(K**((1-alpha)*(X_1-1)))
    expo = 1/(X_1*w+alpha*w-w-X_1*alpha)
    phi_t = X #todo: phi(z_pobre)/phi(z_rico)
    return (L_star*(phi_t**(X_1+alpha-1)))**(expo)    

# Função produto 
def y(M,L,K):
    return (M*K)**(1-alpha)*(L**alpha)

# Função Lucro 
def profit(Y):
    return (1-alpha)*(1-(1/X_1))*Y

# Função salário 
def wage(L):
    return (L)**(w-1)

# Função de choque de saúde 
def z_shock(rho,z,norm_shock):
    if norm_shock == True:
        #Todo: random seed 
        e=0
        e = np.exp(e)
    else:
        e=0 
    return rho*z + float(e)

# Função de consumo 
def consumption(Y,M,K):
    return Y - X_0*(((M)**X_1)/X_1)*K

# Função de taxação 
def Tax(W,L):
    return tau*W*L

# Função de transferencia de recursos para saúde 
def H_pub(Tax,Prices,gamma2):
    return sigma*Tax / (gamma2*Prices)

# Criando a função população 
def population(n):
    dicts={} 
    for i in range(n):
        dicts[str(i)] = {}
        
        # Cidade do individuo
        dicts[str(i)]['city'] = np.random.choice(['0','1','2','3','4'],p=[0.2,0.2,0.2,0.2,0.2])
        
        # Renda do individuo 
        dicts[str(i)]['Income'] = np.random.choice(['0','1'],p=[0.6,0.4]) # 0 = pobre e 1 = rico

        # Taxa de recuperação do individuo 
        dicts[str(i)]['rho'] =1, #float(np.random.choice(['0.8','0.5','0.3','0.0'],p=[0.25,0.25,0.25,0.25]))

        # Consumo do individuo 
        dicts[str(i)]['consumption'] =[]

        # Trabalho do individuo
        dicts[str(i)]['labor'] = []

        # Saúde do individuo = []
        dicts[str(i)]['health'] =[]

        # Choque de saúde 
        dicts[str(i)]['z'] = [0]
        
        # Tecnologia 
        dicts[str(i)]['m'] = []
        
        # Produto 
        dicts[str(i)]['income'] = [] 

        # Lucros 
        dicts[str(i)]['phi'] = []

        # Salários
        dicts[str(i)]['wage'] = []

    return dicts

# Criando a função cidade
def city(n:int,k:list):
    dicts={}

    for i in range(n):
        dicts[str(i)] = {}
        # Valor do Capital 
        dicts[str(i)]['K'] = k[i]

        # Valor da tecnologia
        dicts[str(i)]['m'] = []
        
        # Trabalho Agregado
        dicts[str(i)]['L'] = []

        # Produto Agregado
        dicts[str(i)]['income'] = []
        
        # Lucro Agregado
        dicts[str(i)]['profits'] = []

        # Salário Agregado 
        dicts[str(i)]['wage'] = []

        # Consumo Agregado
        dicts[str(i)]['consumption'] = []

        # Preço de bens de saúde 
        dicts[str(i)]['Prices'] = []

        # Saúde media da cidade 
        dicts[str(i)]['Health'] =[]

        # Individuos da cidade 
        dicts[str(i)]['People'] = []

        # Impostos pagos 
        dicts[str(i)]['Tax'] = []
        
        # Transferencia para saúde pública  
        dicts[str(i)]['h_pub'] =[]

        # Desigualdade do consumo dos demais de bens  
        dicts[str(i)]['Omega_consumption'] = [] 

        # Desigualdade do consumo dos bens de saúde 
        dicts[str(i)]['Omega_health'] = []

        # Desigualdade do consumo total 
        dicts[str(i)]['Omega_total'] = []

        # Classes sociais 
        for j in ['Poors','Richs']:

            # Classes sociais da cidade 
            dicts[str(i)][j] = {}

            # População de cada classe social 
            dicts[str(i)][j]['Population'] =[]

            # Saúde por classe social 
            dicts[str(i)][j]['health'] = [] 

            # Consumo por classe social 
            dicts[str(i)][j]['consumption'] =[] 

    return dicts 

# Criando os individuos 
pop = population(50)

# Criando as cidades 
town = city(n=5,k=[6.04,5.04,4.00,3.01,7.01])

# Identificando os individuo de cada cidade 
for people in range(len(pop)):

    # Cidade do individuo 
    cidade = pop[str(people)]['city']

    # Salvando o individuo na cidade de origem
    city = town[cidade]['People'].append(people)

    # Salvando a classe social do individuo na sua cidade 
    match pop[str(people)]['Income']:
        case '0': 
            town[cidade]['Poors']['Population'].append(people)
        case '1':
            town[cidade]['Richs']['Population'].append(people)

# Calculando os parametros do modelo 
# Calculando os periodos de tempo 
for time in range(100):
    
    # Calculando os parametros de cada individuo 
    for ind in range(len(pop)):
    
        # Capturando o choque de saúde 
        if time == 0:
            shock = True
        else:
            shock = False

        # Calculando a tecnologia 
        #pop[str(ind)]['m'].append(m(pop[str(ind)]['z'][time]))

        # Calculando o trabalho 
        #pop[str(ind)]['labor'].append(L(pop[str(ind)]['z'][time]))

        # Calculando o choque
        pop[str(ind)]['z'].append(z_shock(pop[str(ind)]['rho'],pop[str(ind)]['z'][time],norm_shock=shock))

        # Calculando o phi 
        pop[str(ind)]['phi'].append(phi(pop[str(ind)]['z'][time]))

        # Calculando o consumo 
        #pop[str(ind)]['consumption'].append(consumption(pop[str(ind)]['z'][time]))
    for cid in range(len(town)):

        # Filtrando os vetores de choque por classe social 
        Poors_Phi = [pop[str(i)]['phi'][time] for i in town[str(cid)]['Poors']['Population']]
        Richs_Phi = [pop[str(i)]['phi'][time] for i in town[str(cid)]['Richs']['Population']]

        # Calculando o valor de X 
        X_t = X(poor=Poors_Phi,rich=Richs_Phi,K=town[str(cid)]['K'])
        
        # Calculando m 
        town[str(cid)]['m'].append(m(X_t,K=town[str(cid)]['K']))

        # Calculando o trabalho
        town[str(cid)]['L'].append(L(X_t,K=town[str(cid)]['K']))

        # Calculando o produto 
        town[str(cid)]['income'].append(y(M=town[str(cid)]['m'][time],L=town[str(cid)]['L'][time],K=town[str(cid)]['K']))

        # Calculando o Lucro 
        town[str(cid)]['profits'].append(y(M=town[str(cid)]['m'][time],L=town[str(cid)]['m'][time],K=town[str(cid)]['K']))

        # Calculando os salários 
        town[str(cid)]['wage'].append(wage(town[str(cid)]['L'][time]))

        # Calculando o consumo 
        town[str(cid)]['consumption'].append(consumption(M=town[str(cid)]['m'][time],Y=town[str(cid)]['income'][time],K=town[str(cid)]['K']))

        # Calculando a saúde média dos individuos 
        z_bar = np.mean([pop[str(i)]['z'][time] for i in town[str(cid)]['People']])
        town[str(cid)]['Health'].append(h_bar(m=town[str(cid)]['m'][time],z=z_bar,K=town[str(cid)]['K']))

        # Calculando o preço de bens não saúde 
        town[str(cid)]['Prices'].append(prices(C=town[str(cid)]['consumption'][time],L=town[str(cid)]['L'][time],h_bar=town[str(cid)]['Health'][time]))

        # Calculando a taxação  
        town[str(cid)]['Tax'].append(Tax(W=town[str(cid)]['wage'][time],L=town[str(cid)]['L'][time]))

        # Calculando h_pub 
        town[str(cid)]['h_pub'].append(H_pub(Tax=town[str(cid)]['Tax'][time],Prices=town[str(cid)]['Prices'][time],gamma2=len(town[str(cid)]['Poors']['Population'])/len(town[str(cid)]['People'])))

        # Calculando a saúde dos mais pobres 
        town[str(cid)]['Poors']['health'].append(h_social(prices=town[str(cid)]['Prices'][time],L=town[str(cid)]['L'][time],h_pub=town[str(cid)]['h_pub'][time],h_bar=town[str(cid)]['Health'][time],social='Poors',Pi=0,gamma1=0))

        # Calculando o consumo dos mais pobres 
        town[str(cid)]['Poors']['consumption'].append(c_social(prices=town[str(cid)]['Prices'][time],L=town[str(cid)]['L'][time],h_pub=town[str(cid)]['h_pub'][time],h_bar=town[str(cid)]['Health'][time],social='Poors',Pi=0,gamma1=0))
        
        # Calculado a saúde dos mais ricos
        town[str(cid)]['Richs']['health'].append(h_social(prices=town[str(cid)]['Prices'][time],L=town[str(cid)]['L'][time],h_pub=town[str(cid)]['h_pub'][time],h_bar=town[str(cid)]['Health'][time],social='Richs',Pi=town[str(cid)]['profits'][time],gamma1=len(town[str(cid)]['Richs']['Population'])/len(town[str(cid)]['People'])))

        # Calculando o consumo dos mais ricos 
        town[str(cid)]['Richs']['consumption'].append(c_social(prices=town[str(cid)]['Prices'][time],L=town[str(cid)]['L'][time],h_pub=town[str(cid)]['h_pub'][time],h_bar=town[str(cid)]['Health'][time],social='Richs',Pi=town[str(cid)]['profits'][time],gamma1=len(town[str(cid)]['Richs']['Population'])/len(town[str(cid)]['People'])))

        # Calculando a desigualdade do consumo de bens não saude 
        town[str(cid)]['Omega_consumption'].append(town[str(cid)]['Richs']['consumption'][time]/town[str(cid)]['Poors']['consumption'][time])

        # Calculando a desigualdade do consumo de bens de saúde 
        town[str(cid)]['Omega_health'].append(town[str(cid)]['Richs']['health'][time]/town[str(cid)]['Poors']['health'][time])

        # Calculando a desigualdade do consumo total 
        town[str(cid)]['Omega_total'].append((town[str(cid)]['Poors']['consumption'][time]+town[str(cid)]['Poors']['health'][time])/(town[str(cid)]['Richs']['consumption'][time]+town[str(cid)]['Richs']['health'][time]))




# Gerando os Graficos 
# Choque de produtividade 
plt.plot(town['0']['m'])
plt.plot(town['1']['m'])
plt.legends(['0','1'])
plt.show()

# Choque da oferta de trabalho
plt.plot(town['0']['L'])
plt.plot(town['1']['L'])
plt.legends(['0','1'])
plt.show()

# Choque do produto 
plt.plot(town['0']['income'])
plt.plot(town['1']['income'])
plt.legends(['0','1'])
plt.show()

