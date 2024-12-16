# %% Importando calculos estatisticos
import inferencia_estatistica as ie
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import f_oneway

# %%
# Dados de Estudantes Gravidas
mulheres_gravidas_2015 = np.array([6.4, 6.8,11.3,13.3,6.3,14.0,7.5,15.4,7.6,6.2,8.8,10.2, 9.6,12.5,10.0, 6.2,6.4, 5.9,9.6,4.9, 4.6, 2.9,5.7, 4.2,7.3, 4.1, 4.0])/100
mulheres_gravidas_2019 = np.array([5.9,14.6, 0.0, 4.9,5.6, 6.4,5.1, 0.8,0.9,6.4,1.7,12.8,12.9,17.3,14.8,17.3,2.0,10.1,7.4,2.0,15.8,12.6,6.7,16.5,5.4,11.7,12.3])/100

capitais = ['Porto Velho', 'Rio Branco', 'Manaus', 'Boa Vista', 'Belém', 'Macapá', 'Palmas', 'São Luís', 'Teresina', 'Fortaleza', 'Natal', 'João Pessoa', 'Recife', 'Maceió', 'Aracaju', 'Salvador', 'Belo Horizonte', 'Vitória', 'Rio de Janeiro', 'São Paulo', 'Curitiba', 'Florianópolis', 'Porto Alegre', 'Campo Grande', 'Cuiabá', 'Goiânia', 'Brasília']

df_mulheres_gravidas_2015 = pd.DataFrame({
    'capitais': capitais,
    'ano': 2015,
    'percentual_gravida': mulheres_gravidas_2015,
})

df_mulheres_gravidas_2019 = pd.DataFrame({
    'capitais': capitais,
    'ano': 2019,
    'percentual_gravida': mulheres_gravidas_2019,
})

df_mulheres_gravidas = pd.concat([df_mulheres_gravidas_2015, df_mulheres_gravidas_2019])
# %%
# 2015
mediana_2015 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2015]['percentual_gravida'].median()
media_2015 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2015]['percentual_gravida'].mean()
var_2015 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2015]['percentual_gravida'].var()

print('Gravidas: Medidas descritivas 2015')
print(f'Media:{media_2015}')
print(f'Mediana:{mediana_2015}')
print(f'Variancia:{var_2015}')

# 2019
mediana_2019 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2019]['percentual_gravida'].median()
media_2019 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2019]['percentual_gravida'].mean()
var_2019 = df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2019]['percentual_gravida'].var()

print('Gravidas: Medidas descritivas 2019')
print(f'Media:{media_2019}')
print(f'Mediana:{mediana_2019}')
print(f'Variancia:{var_2019}')


# %%

plt.figure(figsize=(12,6))
sns.boxplot(x='ano', y='percentual_gravida', data=df_mulheres_gravidas, palette='Set2')
plt.scatter(x=[0], y=[media_2015], color='red', zorder=5, label=f'Média 2015: {media_2015:.2f}')
plt.scatter(x=[1], y=[media_2019], color='blue', zorder=5, label=f'Média 2019: {media_2019:.2f}')

plt.title('Boxplot taxa de gravidez', fontsize=16, weight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Percentuais', fontsize=14)
plt.legend()
plt.show()
# %%
# Supondo que os dados estejam no DataFrame `df`
# plt.figure(figsize=(10, 6))
# sns.scatterplot(data=df_mulheres_gravidas, x="capitais", y="percentual_gravida", marker="o")
# plt.title("Tendência da taxa de gravidez (2015-2019)")
# plt.xlabel("Ano")
# plt.ylabel("Percentual de Escolares com Internet")
# plt.legend(title="Grupo")
# plt.grid()
# plt.show()

# %%
estat, p = ie.shapiro(df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2015]['percentual_gravida'])
print('Teste de Normalidade Shapiro-Wilk para os dados de taxa de gravidez em 2015')
print(f'Estatistica W: {estat}')
print(f'Siginificancia: {p}')
# %%
estat, p = ie.shapiro(df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2019]['percentual_gravida'])
print('Teste de Normalidade Shapiro-Wilk para os dados de taxa de gravidez em 2019')
print(f'Estatistica W: {estat}')
print(f'Siginificancia: {p}')
# %%
# Teste de variancia
#2009 com 2012
r, est_th_var = ie.th_variancia_amostra_dependentes(var_amostral1=var_2015, var_amostral2=var_2019, n1=27, n2=27, hipotese_alternativa='!=', alfa=0.05)

print('Teste de Hipótese para a variancia Mulheres Gravidas 2015 e 2019')
print(r)
print(f'Estatistica Calculada: {est_th_var[0]}')
print(f'Estaticas Critica tabelada: {est_th_var[1]} e {est_th_var[2]}')

# %%
# Teste de hipotese para a média com amostras dependentes e heterocedásticas
r, est_th_var = ie.th_media_amostra_dependentes_var_pop_desconhecida_heterocedastica(media_2015, media_2019, var_2015, var_2019, 27, 27, 0.05, '!=')

print('Teste de Hipótese para a variancia Mulheres Gravidas 2015 e 2019')
print(r)
print(f'Estatistica Calculada: {est_th_var[0]}')
print(f'Estaticas Critica tabelada: {est_th_var[1]} e {est_th_var[2]}')

# %%
ie.wilcoxon(df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2015]['percentual_gravida'], df_mulheres_gravidas[df_mulheres_gravidas['ano'] == 2019]['percentual_gravida'], alternative='greater')



# %%
# Dados de estudantes que receberam orientação de prevenção a gravidez
mulheres_orientadas_2015 = np.array([78.8,88.0,79.0,78.1,65.4,78.1,81.0,79.7,75.9,76.4,78.4,78.7,69.5,79.7,81.1,73.7,81.2,80.7,73.5,81.7,82.0,82.5,76.1,78.6,75.1,80.3,80.8])/100
mulheres_orientadas_2019 = np.array([82.7,90.5,93.6,70.6,83.9,73.9,66.1,79.5,72.9,74.8,82.7,77.5,71.8,79.2,84.8,80.0,76.9,85.8,73.6,77.3,85.4,89.0,76.3,71.6,84.6,77.3,80.1])/100

capitais = ['Porto Velho', 'Rio Branco', 'Manaus', 'Boa Vista', 'Belém', 'Macapá', 'Palmas', 'São Luís', 'Teresina', 'Fortaleza', 'Natal', 'João Pessoa', 'Recife', 'Maceió', 'Aracaju', 'Salvador', 'Belo Horizonte', 'Vitória', 'Rio de Janeiro', 'São Paulo', 'Curitiba', 'Florianópolis', 'Porto Alegre', 'Campo Grande', 'Cuiabá', 'Goiânia', 'Brasília']

df_mulheres_orientadas_2015 = pd.DataFrame({
    'capitais': capitais,
    'ano': 2015,
    'percentual_orientada': mulheres_orientadas_2015,
})

df_mulheres_orientadas_2019 = pd.DataFrame({
    'capitais': capitais,
    'ano': 2019,
    'percentual_orientada': mulheres_orientadas_2019,
})

df_mulheres_orientadas = pd.concat([df_mulheres_orientadas_2015, df_mulheres_orientadas_2019])
# %%
# 2015
mediana_2015 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'].median()
media_2015 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'].mean()
var_2015 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'].var()

print('Orientadas: Medidas descritivas 2015')
print(f'Media:{media_2015}')
print(f'Mediana:{mediana_2015}')
print(f'Variancia:{var_2015}')

# 2019
mediana_2019 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'].median()
media_2019 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'].mean()
var_2019 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'].var()
print('Orientadas: Medidas descritivas 2019')
print(f'Media:{media_2019}')
print(f'Mediana:{mediana_2019}')
print(f'Variancia:{var_2019}')

# %%
media_2015 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'].mean()
media_2019 = df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'].mean()

plt.figure(figsize=(12,6))
sns.boxplot(x='ano', y='percentual_orientada', data=df_mulheres_orientadas, palette='Set2')
plt.scatter(x=[0], y=[media_2015], color='red', zorder=5, label=f'Média 2015: {media_2015:.2f}')
plt.scatter(x=[1], y=[media_2019], color='blue', zorder=5, label=f'Média 2019: {media_2019:.2f}')

plt.title('Boxplot da Taxa de Mulheres orientadas', fontsize=16, weight='bold')
plt.xlabel('Ano', fontsize=14)
plt.ylabel('Percentuais', fontsize=14)
plt.legend()
plt.show()
# %%
# Supondo que os dados estejam no DataFrame `df`
# plt.figure(figsize=(10, 6))
# sns.lineplot(data=df_mulheres_orientadas, x="ano", y="percentual_orientada", marker="o")
# plt.title("Tendência da taxa de mulheres orientadas (2015-2019)")
# plt.xlabel("Ano")
# plt.ylabel("Percentual de Escolares com Internet")
# plt.legend(title="Grupo")
# plt.grid()
# plt.show()

# %%
estat, p = ie.shapiro(df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'])
print('Teste de Normalidade Shapiro-Wilk para os dados de mulheres orientadas em 2015')
print(f'Estatistica W: {estat}')
print(f'Siginificancia: {p}')
# %%
estat, p = ie.shapiro(df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'])
print('Teste de Normalidade Shapiro-Wilk para os dados de mulheres orientadas em 2019')
print(f'Estatistica W: {estat}')
print(f'Siginificancia: {p}')
# %%
# Teste de variancia
#2009 com 2012
r, est_th_var = ie.th_variancia_amostra_dependentes(var_amostral1=var_2015, var_amostral2=var_2019, n1=27, n2=27, hipotese_alternativa='!=', alfa=0.05)

print('Teste de Hipótese para a variancia Mulheres Orientadas 2015 e 2019')
print(r)
print(f'Estatistica Calculada: {est_th_var[0]}')
print(f'Estaticas Critica tabelada: {est_th_var[1]} e {est_th_var[2]}')
# %%
# Teste de hipotese para a média com amostras dependentes e heterocedásticas
r, est_th_var = ie.th_media_amostra_dependentes_var_pop_desconhecida_heterocedastica(media_2015, media_2019, var_2015, var_2019, 27, 27, 0.05, '!=')

print('Teste de Hipótese para a media Mulheres Orientadas 2015 e 2019')
print(r)
print(f'Estatistica Calculada: {est_th_var[0]}')
print(f'Estaticas Critica tabelada: {est_th_var[1]} e {est_th_var[2]}')

# %%
ie.wilcoxon(df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2015]['percentual_orientada'], df_mulheres_orientadas[df_mulheres_orientadas['ano'] == 2019]['percentual_orientada'], alternative='greater')
# %%
ie.coef_corr_simples(df_mulheres_orientadas['percentual_orientada'], df_mulheres_gravidas['percentual_gravida'])

# %%
