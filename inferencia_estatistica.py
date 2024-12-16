#%%
import pandas as pd
import numpy as np
from scipy.stats import norm, t, chi2, f, binomtest, kstest, shapiro, wilcoxon, mannwhitneyu, kruskal, spearmanr, f_oneway 
import matplotlib.pyplot as plt
import seaborn as sns
#%% Logica de decisão da Hipótese
# Logica de decisão da Hipótese
def decisao_hipotese_z(z_hipotese, alfa, hipotese_alternativa):
    if hipotese_alternativa == "!=":
        z_critico_inferior = round(norm.ppf(alfa), 3)
        z_critico_superior = round(norm.ppf(1-alfa), 3)
        if z_critico_inferior < z_hipotese < z_critico_superior:
            resultado = 'Hipotese aceita'
            return resultado, [z_hipotese, z_critico_inferior, z_critico_superior]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [z_hipotese, z_critico_inferior, z_critico_superior]
    elif hipotese_alternativa == "<":
        z_critico = round(norm.ppf(alfa), 3)
        if z_critico < z_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [z_hipotese, z_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [z_hipotese, z_critico]
    elif hipotese_alternativa == ">":
        z_critico = round(norm.ppf(alfa), 3)
        if z_critico > z_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [z_hipotese, z_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [z_hipotese, z_critico]

def decisao_hipotese_t(t_hipotese, alfa, grau_de_liberdade, hipotese_alternativa):
    if hipotese_alternativa == "!=":
        t_critico_inferior = round(t.ppf(alfa, grau_de_liberdade), 3)
        t_critico_superior = round(t.ppf(1-alfa, grau_de_liberdade), 3)
        if t_critico_inferior < t_hipotese < t_critico_superior:
            resultado = 'Hipotese aceita'
            return resultado, [t_hipotese, t_critico_inferior, t_critico_superior]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [t_hipotese, t_critico_inferior, t_critico_superior]
    elif hipotese_alternativa == "<":
        t_critico = round(t.ppf(alfa, grau_de_liberdade), 3)
        if t_critico < t_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [t_hipotese, t_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [t_hipotese, t_critico]
    elif hipotese_alternativa == ">":
        t_critico = round(t.ppf(alfa, grau_de_liberdade), 3)
        if t_critico > t_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [t_hipotese, t_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [t_hipotese, t_critico]

def decisao_hipotese_chi2(chi2_hipotese, alfa, grau_de_liberdade, hipotese_alternativa):
    if hipotese_alternativa == "!=":
        chi2_critico_inferior = round(chi2.ppf(alfa, grau_de_liberdade), 3)
        chi2_critico_superior = round(chi2.ppf(1-alfa, grau_de_liberdade), 3)
        if chi2_critico_inferior < chi2_hipotese < chi2_critico_superior:
            resultado = 'Hipotese aceita'
            return resultado, [chi2_hipotese, chi2_critico_inferior, chi2_critico_superior]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [chi2_hipotese, chi2_critico_inferior, chi2_critico_superior]
    elif hipotese_alternativa == "<":
        chi2_critico = round(chi2.ppf(alfa, grau_de_liberdade), 3)
        if chi2_critico < chi2_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [chi2_hipotese, chi2_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [chi2_hipotese, chi2_critico]
    elif hipotese_alternativa == ">":
        chi2_critico = round(chi2.ppf(alfa, grau_de_liberdade), 3)
        if chi2_critico > chi2_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [chi2_hipotese, chi2_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [chi2_hipotese, chi2_critico]

def decisao_hipotese_f(f_hipotese, alfa, n1, n2, hipotese_alternativa):
    if hipotese_alternativa == "!=":
        f_critico_inferior = round(f.ppf(alfa, n1, n2), 3)
        f_critico_superior = round(f.ppf(1-alfa, n1, n2), 3)
        if f_critico_inferior < f_hipotese < f_critico_superior:
            resultado = 'Hipotese aceita'
            return resultado, [f_hipotese, f_critico_inferior, f_critico_superior]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [f_hipotese, f_critico_inferior, f_critico_superior]
    elif hipotese_alternativa == "<":
        f_critico = round(f.ppf(alfa, n1, n2), 3)
        if f_critico < f_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [f_hipotese, f_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [f_hipotese, f_critico]
    elif hipotese_alternativa == ">":
        f_critico = round(f.ppf(alfa, n1, n2), 3)
        if f_critico > f_hipotese:
            resultado = 'Hipotese aceita'
            return resultado, [f_hipotese, f_critico]
        else:
            resultado = 'Hipotese Rejeitada'
            return resultado, [f_hipotese, f_critico]

# %% Intervalo de confianca - média com variancia conhecida
# Intervalo de confianca - média com variancia conhecida
def ic_media_variancia_conhecida(media_amostral, variancia_populacional, n, alfa= 0.025):
    margem = norm.ppf(alfa)*(np.sqrt(variancia_populacional)/ np.sqrt(n))
    ic_superior = media_amostral + margem
    ic_inferior = media_amostral - margem
    return ic_inferior, ic_superior

# %% Intervalo de confianca - média com variancia desconhecida
# Intervalo de confianca - média com variancia desconhecida
def ic_media_variancia_desconhecida(media_amostral, variancia_amostral, n, alfa= 0.025):
    grau_de_liberdade = n-1
    margem = t.ppf(alfa, grau_de_liberdade)*(np.sqrt(variancia_amostral)/ np.sqrt(n))
    ic_superior = media_amostral + margem
    ic_inferior = media_amostral - margem
    return ic_inferior, ic_superior

# %% Intervalo de Confianca - variancia da populacao
# Intervalo de Confianca - variancia da populacao
def ic_variancia_populacao(n, variancia_amostral, alfa):
    grau_de_liberdade = n-1
    xb = chi2.ppf(1-alfa, grau_de_liberdade)
    xa = chi2.ppf(alfa, grau_de_liberdade)
    ic_superior = ((n-1)*variancia_amostral)/xa
    ic_inferior = ((n-1)*variancia_amostral)/xb
    return ic_inferior, ic_superior

# %% Intervalo de confianca - Proporcao da populacao
# Intervalo de confianca - Proporcao da populacao
def ic_proporcao_populacao(n, x_alvo, alfa):
    freq = x_alvo / n
    margem = norm.ppf(alfa) * np.sqrt((freq*(1-freq)) / n)
    ic_inferior = freq - margem
    ic_superior = freq + margem
    return ic_superior, ic_inferior

# %% Teste de Hipotese para a media com variancia conhecida
# Teste de Hipotese para a media com variancia conhecida
def th_media_variancia_conhecida(media_amostral, hipotese_nula, variancia_populacional, n, alfa, hipotese_alternativa):
    z_hipotese = round((media_amostral - hipotese_nula) / (np.sqrt(variancia_populacional) /np.sqrt(n)), 3)
    result, list_z = decisao_hipotese_z(z_hipotese=z_hipotese, alfa=alfa, hipotese_alternativa=hipotese_alternativa)
    return result, list_z

# %% Teste de Hipotese para a media com variancia desconhecida
# Teste de Hipotese para a media com variancia desconhecida
def th_media_variancia_desconhecida(media_amostral, hipotese_nula, variancia_amostral, n, alfa, hipotese_alternativa):
    grau_de_liberdade = n-1
    t_hipotese = round((media_amostral - hipotese_nula) / (np.sqrt(variancia_amostral) /np.sqrt(n)), 3)
    result, t_list = decisao_hipotese_t(t_hipotese=t_hipotese, grau_de_liberdade=grau_de_liberdade, alfa=alfa, hipotese_alternativa=hipotese_alternativa)
    return result, t_list

# %% Teste de Hipotese para a proporcao
# Teste de Hipotese para a proporcao
def th_proporcao(hipotese_nula, observacoes, n, alfa, hipotese_alternativa):
    frequencia = observacoes / n
    z_hipotese = round(((frequencia - hipotese_nula) / np.sqrt((hipotese_nula*(1-hipotese_nula)/n))), 3)
    result, list_z = decisao_hipotese_z(z_hipotese=z_hipotese, alfa=alfa, hipotese_alternativa=hipotese_alternativa)
    return result, list_z

# %% Teste de Hipotese para variancia
# Teste de Hipotese para variancia
def th_variancia(hipotese_nula, variancia_amostral, n, alfa, hipotese_alternativa):
    grau_de_liberdade = n-1
    chi2_hipotese = round((((n-1)*variancia_amostral)/hipotese_nula), 3)
    result, list_chi2 = decisao_hipotese_chi2(chi2_hipotese=chi2_hipotese, alfa=alfa, grau_de_liberdade=grau_de_liberdade, hipotese_alternativa=hipotese_alternativa)
    return result, list_chi2

# %% Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional conhecida
# Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional conhecida
def th_media_amostra_dependentes_var_pop_conhecida(media_amostral1, media_amostral2, var_pop1, var_pop2, n1, n2, alfa, hipotese_alternativa):
    z_hipotese = (media_amostral1 - media_amostral2) / np.sqrt((var_pop1/n1)+(var_pop2/n2))
    result, list_z = decisao_hipotese_z(z_hipotese=z_hipotese, alfa=alfa, hipotese_alternativa=hipotese_alternativa)
    return result, list_z

# %% Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional descconhecida Homocedastica
# Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional descconhecida Homocedastica
def th_media_amostra_dependentes_var_pop_desconhecida_homocedastica(media_amostral1, media_amostral2, var_amostral1, var_amostral2, n1, n2, alfa, hipotese_alternativa):
    grau_de_liberdade = n1+n2-2
    sc = np.sqrt( (((n1-1)*var_amostral1) + (n2-1)*var_amostral2) / (n1+n2-2))
    t_hipotese = (media_amostral1-media_amostral2) / (sc * np.sqrt((n1+n2) / (n1*n2)))
    result, list_t = decisao_hipotese_t(t_hipotese=t_hipotese, alfa=alfa, grau_de_liberdade=grau_de_liberdade, hipotese_alternativa=hipotese_alternativa)
    return result, list_t

#%% Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional descconhecida Heterocedastica
# Teste de Hipotese Media Populacional - Amostras dependentes com variancia populacional descconhecida Heterocedastica
def th_media_amostra_dependentes_var_pop_desconhecida_heterocedastica(media_amostral1, media_amostral2, var_amostral1, var_amostral2, n1, n2, alfa, hipotese_alternativa):
    grau_de_liberdade = (((var_amostral1/n1) + (var_amostral2/n2))**2) / ((((var_amostral1/n1)**2)/(n1-1)) + (((var_amostral2/n2)**2)/(n2-1)))
    t_hipotese = (media_amostral1-media_amostral2) / np.sqrt((var_amostral1/n1) + (var_amostral2/n2))
    result, list_t = decisao_hipotese_t(t_hipotese=t_hipotese, alfa=alfa, grau_de_liberdade=grau_de_liberdade, hipotese_alternativa=hipotese_alternativa)
    return result, list_t

# %% Teste de Hipotese Variancia amostras dependentes
# Teste de Hipotese Variancia amostras dependentes
def th_variancia_amostra_dependentes(var_amostral1, var_amostral2, n1, n2, hipotese_alternativa, alfa):
    n1 = n1-1
    n2 = n2-1
    f_hipotese = var_amostral1 / var_amostral2
    result, list_f = decisao_hipotese_f(f_hipotese=f_hipotese, alfa=alfa, n1=n1, n2=n2, hipotese_alternativa=hipotese_alternativa)
    return result, list_f
# %% Teste de Hipotese Proporcao Populacional - Amostras Depentes
# Teste de Hipotese Proporcao Populacional - Amostras Depentes
def th_proporcao_amostras_dependentes(obs1, obs2, n1, n2, alfa, hipotese_alternativa):
    p = (obs1+obs2) / (n1+n2)
    f1 = obs1/n1
    f2 = obs2/n2
    z_hipotese = (f1 - f2) / np.sqrt(p*(1-p)*((1/n1)+(1/n2)))
    result, list_z = decisao_hipotese_z(z_hipotese=z_hipotese, alfa=alfa, hipotese_alternativa=hipotese_alternativa)
    return result, list_z

# %% Teste de Aderencia Chi_quadrado
# Teste de Aderencia Chi_quadrado
def taderencia_chi2(freq_observada, freq_esperada, k, alfa):
    phi = k-1
    chi2_hipotese = 0
    for i in range(len(freq_observada)):
        chi2_hipotese += ((freq_observada[i] - freq_esperada[i])**2) / freq_esperada[i]
    result, list_chi2 = decisao_hipotese_chi2(chi2_hipotese=chi2_hipotese, alfa=alfa, grau_de_liberdade=phi, hipotese_alternativa='>')
    return result, list_chi2

# %% Teste de Independencia Chi-quadrado
# Teste de Independencia Chi-quadrado
def tindependencia_chi2(tabela_contingencia, alfa):
    phi = (len(tabela_contingencia)-1)*(len(tabela_contingencia[0])-1)
    chi2_hipotese = 0
    soma_linhas = [sum(linha) for linha in tabela_contingencia]
    soma_colunas = [sum(coluna) for coluna in zip(*tabela_contingencia)]
    for l in range(len(tabela_contingencia)):
        for c in range(len(tabela_contingencia[l])):
            soma_linha = soma_linhas[l]
            soma_coluna = soma_colunas[c]
            feij = (soma_linha * soma_coluna) / sum(soma_linhas)
            chi2_hipotese += ((tabela_contingencia[l][c]-feij)**2)/feij
    result, list_chi2 = decisao_hipotese_chi2(chi2_hipotese=chi2_hipotese, alfa=alfa, grau_de_liberdade=phi, hipotese_alternativa='>')
    return result, list_chi2


tcont = [
    [25, 35],
    [15, 25],
    [10, 30],
    [50, 90]
]
alfa = 0.95

tindependencia_chi2(tcont, alfa)

# %% Teste de Normalidade Kolmogorov-Smirnov
# Teste de Normalidade Kolmogorov-Smirnov - média e desvio padrão populacional conhecidos
# dados = [
#     2.2, 4.1, 3.5, 4.5, 5.0, 3.7, 3.0, 2.6, 3.4, 1.6,
#     3.1, 3.3, 3.8, 3.1, 4.7, 3.7, 2.5, 4.3, 4.9, 3.6,
#     2.9, 3.3, 3.9, 3.1, 4.8, 3.1, 3.7, 4.4, 3.2, 4.1,
#     1.9, 3.4, 4.7, 3.8, 3.0, 2.6, 3.9, 3.0, 4.2, 3.5
# ]

# kstest(rvs=dados, cdf='norm', args=(np.mean(dados),np.std(dados, ddof=1)))
# %% Teste de Normalidade Shapiro-Wilk
# Teste de Normalidade Shapiro-Wilk
# dados = [1.90642, 2.10288, 1.52229, 2.61826, 1.42738, 2.22488, 1.69742, 3.15435, 1.98492, 1.99568]
# shapiro(dados)
# if p_valor > 0.05:
#     print("Dados considerados normalmente distribuidos com 0.05")
# %% Teste dos Sinais
def teste_sinais(n_positivos, n, p, alfa=0.05, hip_alt="!="):
    if n > 20:
        z = (n_positivos - (n*p)) / np.sqrt(n*p*(1-p))
        result = decisao_hipotese_z(z, alfa, hip_alt)
    else:
        if hip_alt == '!=':
            result = binomtest(n_positivos, n, p, alternative='two-sided')
        elif hip_alt == ">":
            result = binomtest(n_positivos, n, p, alternative='greater')
        elif hip_alt == "<":
            result = binomtest(n_positivos, n, p, alternative='less')
    return result

def tabela_sinais(antes, depois):
    sinais = []
    for i in range(len(antes)):
        if antes[i] > depois[i]:
            sinais.append(-1)
        elif antes[i] < depois[i]:
            sinais.append(1)
        else:
            sinais.append(0)
    n_pos = sinais.count(1)
    n_neg = sinais.count(-1)
    return n_pos, n_neg


# antes =  [55, 63, 78, 81, 68, 58, 60, 60, 48, 49, 90, 93, 90, 56, 66, 75, 85, 90, 50, 58, 83, 47, 73, 74, 48, 68, 72, 86, 80, 67]
# depois = [50, 65, 78, 79, 70, 57, 58, 62, 50, 51, 81, 85, 90, 58, 64, 70, 81, 80, 60, 55, 75, 52, 70, 70, 53, 65, 70, 83, 81, 68]

# pos, neg = tabela_sinais(antes, depois)
# teste_sinais(pos, pos+neg, 0.5, 0.025, "<")

# %% Teste de Wilcoxon
# antes =  [30, 19, 19, 23, 29, 17, 42, 20, 12, 39, 14, 81, 17, 31, 52]
# depois = [30, 14,  8, 14, 52, 14, 22, 17,  8, 11, 30, 14, 17, 15, 43]


# wilcoxon(antes, depois)
# %% Teste de mann-whitney
def teste_mann_whiteney(a1, a2):
    dados_a1 = pd.DataFrame({
            'dados': a1,
            'grupo': 'A1'
        })
    dados_a2 = pd.DataFrame({
            'dados': a2,
            'grupo': 'A2'
        })

    dados = pd.concat([dados_a1, dados_a2])
    dados['rank'] = dados['dados'].rank()

    r1 = dados['rank'][dados['grupo'] == "A1"].sum()
    n1 = len(dados['rank'][dados['grupo'] == "A1"])
    r2 = dados['rank'][dados['grupo'] == "A2"].sum()
    n2 = len(dados['rank'][dados['grupo'] == "A2"])
    u1 = n1*n2 + ((n1*(n1+1))/2) - r1
    u2 = n1*n2 + ((n2*(n2+1))/2) - r2
    return min([u1, u2])

# amostra_1 = np.array([500, 1200, 2500, 6000, 8000])
# amostra_2 = np.array([600, 750, 1000, 3200, 5000, 7500])

# teste_mann_whiteney(amostra_1, amostra_2)
# mannwhitneyu(amostra_1, amostra_2, alternative='two-sided')
# %% Teste da Mediana
def teste_mediana(a1, a2, alfa):
    dados_a1 = pd.DataFrame({
            'dados': a1,
            'grupo': 'A1'
        })
    dados_a2 = pd.DataFrame({
            'dados': a2,
            'grupo': 'A2'
        })

    dados = pd.concat([dados_a1, dados_a2])
    mediana = dados['dados'].median()

    f_obs_maior_a1 = dados['dados'][(dados['dados'] > mediana) & (dados['grupo'] == 'A1')].count()
    f_obs_menor_a1 = dados['dados'][(dados['dados'] <= mediana) & (dados['grupo'] == 'A1')].count()
    f_obs_maior_a2 = dados['dados'][(dados['dados'] > mediana) & (dados['grupo'] == 'A2')].count()
    f_obs_menor_a2 = dados['dados'][(dados['dados'] <= mediana) & (dados['grupo'] == 'A2')].count()

    f_esp_maior_a1 = ((f_obs_maior_a1 + f_obs_maior_a2) * (f_obs_maior_a1 + f_obs_menor_a1))/ len(dados['dados'])
    f_esp_menor_a1 = ((f_obs_menor_a1 + f_obs_menor_a2) * (f_obs_maior_a1 + f_obs_menor_a1))/ len(dados['dados'])
    f_esp_maior_a2 = ((f_obs_maior_a1 + f_obs_maior_a2) * (f_obs_maior_a2 + f_obs_menor_a2))/ len(dados['dados'])
    f_esp_menor_a2 = ((f_obs_menor_a1 + f_obs_menor_a2) * (f_obs_maior_a2 + f_obs_menor_a2))/ len(dados['dados'])

    chi_calculado = (((f_obs_maior_a1 - f_esp_maior_a1) ** 2)/ f_esp_maior_a1) + (((f_obs_maior_a2 - f_esp_maior_a2) ** 2)/ f_esp_maior_a2) +(((f_obs_menor_a1 - f_esp_menor_a1) ** 2)/ f_esp_menor_a1) +(((f_obs_menor_a2 - f_esp_menor_a2) ** 2)/ f_esp_menor_a2)

    return decisao_hipotese_chi2(chi_calculado, alfa, 1, ">")

# a1 = [8,7,5,5,10,4,6,9,3]
# a2 = [1,2,4,6,8,10,7]

# teste_mediana(a1, a2, 0.95)
# %% Teste Kruskall Wallis?
# Centro  = [13,34,27,39,19,33,25,46,37,17]
# SCrist3 = [22,43,30,14,44,40,48,15,29,42]
# Humaita = [18,41,23,45,32,35,50,16,49,51]
# Tijuca  = [12,36,31,38,20,47,28,52,26,11]

# stat, p_value = kruskal(Centro, SCrist3, Humaita, Tijuca)
# print(f"Estatística H: {stat}")
# print(f"Valor p: {p_value}")
# print(chi2.ppf(0.05, 3))
# %% Teste de Correlação de Spearman
# x = [17, 20, 22, 28, 42, 55, 75, 80, 90, 145, 145, 170]
# y = [42, 40, 30,  7, 12, 10,  7,  3,  7,   5,   2,   4]


# rho, p_value = spearmanr(x, y)
# print(f"Coeficiente de Spearman: {rho}")
# print(f"Valor p: {p_value}")
# %% ANOVA
# baixa=[106,90,103,90,79,88,92,95]
# media=[80 ,69,94 ,91,70,83,87,83]
# alta= [78 ,80,62 ,69,76,85,69,85]

# # Criação do DataFrame
# df = pd.DataFrame({
#     'valores': alta + media + baixa,
#     'caracteristicas': ['alta']*len(alta) + ['media']*len(media) + ['baixa']*len(baixa)
# })

# Dados originais
# laguna1 = [37.54, 37.01, 36.71, 37.03, 37.32, 37.01, 37.03, 37.70, 37.36, 36.75, 37.45, 38.85]
# laguna2 = [40.17, 40.80, 39.76, 39.70, 40.79, 40.44, 39.79, 39.38, 38.51, 40.08]
# laguna3 = [39.04, 39.21, 39.05, 38.24, 38.53, 38.71, 38.89, 38.66]

# # Transformar os dados em um único vetor
# valores = laguna1 + laguna2 + laguna3

# # Criar o vetor de características
# caracteristicas = ['Laguna 1'] * len(laguna1) + ['Laguna 2'] * len(laguna2) + ['Laguna 3'] * len(laguna3)

# # Criar o DataFrame
# df = pd.DataFrame({'valores': valores, 'caracteristicas': caracteristicas})

# A = [64, 72, 68, 77, 56, 95]
# B = [78, 91, 97, 82, 85, 77]
# C = [75, 93, 78, 71, 63, 76]
# D = [55, 66, 49, 64, 70, 68]

# valores = A + B + C + D
# caracteristicas = ['A'] * len(A) + ['B'] * len(B) + ['C'] * len(C) + ['D'] * len(D)

# df = pd.DataFrame({'valores': valores, 'caracteristicas': caracteristicas})


def anova(df, alfa = 0.01):
    sns.boxplot(x='caracteristicas', y='valores', data=df, palette='Set2', hue='caracteristicas')
    plt.title('Boxplot', fontsize=16, weight='bold')
    plt.xlabel('caracteristicas', fontsize=14)
    plt.ylabel('valores', fontsize=14)
    plt.show()

    gl_entre_grupos = len(df['caracteristicas'].unique()) - 1
    gl_dentro_grupos = len(df['valores']) - len(df['caracteristicas'].unique())
    gl_total = len(df['valores']) - 1

    f_crit = f.ppf(1-alfa, gl_entre_grupos, gl_dentro_grupos)

    sq_total = (df['valores']**2).sum() - ((df['valores'].sum())**2 / df['valores'].count())

    sq_trat = 0
    for c in df['caracteristicas'].unique():
        sq_trat += (df[df['caracteristicas'] == c]['valores'].sum())**2 / len(df[df['caracteristicas'] == c]['valores'])
    sq_trat = sq_trat - ((df['valores'].sum())**2 / df['valores'].count())
    sq_trat

    sq_erro = sq_total - sq_trat

    qm_trat = sq_trat / gl_entre_grupos
    qm_erro = sq_erro / gl_dentro_grupos
    f_calc = qm_trat / qm_erro
    if f_calc > f_crit:
        print("Hipotese nula rejeitada")
    else:
        print('Hipotese nula aceita')

#     print(gl_entre_grupos)
#     print(gl_dentro_grupos)
#     print(gl_total)
#     print(f_crit)
#     print(sq_total)
#     print(sq_trat)
#     print(sq_erro)
#     print(qm_trat)
#     print(qm_erro)
#     print(f_calc)

# anova(df)

# %% Coeficiente de correlação simples
x = np.array([6.4, 6.8,11.3,13.3,6.3,14.0,7.5,15.4,7.6,6.2,8.8,10.2, 9.6,12.5,10.0, 6.2,6.4, 5.9,9.6,4.9, 4.6, 2.9,5.7, 4.2,7.3, 4.1, 4.0])/100
y = np.array([78.8,88.0,79.0,78.1,65.4,78.1,81.0,79.7,75.9,76.4,78.4,78.7,69.5,79.7,81.1,73.7,81.2,80.7,73.5,81.7,82.0,82.5,76.1,78.6,75.1,80.3,80.8])/100
n = len(x)
# %%
def coef_corr_simples(x, y):
    n = len(x)
    return ((x*y).sum() - (x.sum()*y.sum()/n)) / np.sqrt(((x**2).sum() - ((x.sum())**2/n)) * ((y**2).sum() - ((y.sum())**2/n)))
# %%
coef_corr_simples(x, y)
# %%
