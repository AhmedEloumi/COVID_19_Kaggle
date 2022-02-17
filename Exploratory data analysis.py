import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_row',111)
pd.set_option('display.max_column',111) #afficher tous les colonnes
covid=pd.read_excel("covid 19.xlsx")

print(covid.head())

"""# 1. Exploratory Data Analysis

## Objectif :
- Comprendre du mieux possible nos données (un petit pas en avant vaut mieux qu'un grand pas en arriere)
- Développer une premiere stratégie de modélisation 

## Checklist de base
#### Analyse de Forme :
- **variable target** : SARS-Cov-2 exam result
- **lignes et colonnes** : 5644, 111
- **types de variables** : qualitatives : 70, quantitatives : 41
- **Analyse des valeurs manquantes** :
    - beaucoup de NaN (moitié des variables > 90% de NaN)
    - 2 groupes de données 76% -> Test viral, 89% -> taux sanguins

#### Analyse de Fond :
- **Visualisation de la target** :
    - 10% de positifs (558 / 5000)
    
    
    
- **Signification des variables** :
    -  variables continues standardisées, skewed (asymétriques), test sanguin
    - age quantile : difficile d'interpreter ce graphique, clairement ces données ont été traitées, on pourrait penser 0-5, mais cela pourrait aussi etre une transformation mathématique. On peut pas savoir car la personne qui a mit ce dataset ne le précise nul part. Mais ca n'est pas tres important
    - variable qualitative : binaire (0, 1), viral, Rhinovirus qui semble tres élevée



- **Relation Variables / Target** :
    - target / blood : les taux de Monocytes, Platelets, Leukocytes semblent liés au covid-19 -> hypothese a tester
    - target/age : les individus de faible age sont tres peu contaminés ? -> attention on ne connait pas l'age, et on ne sait pas de quand date le dataset (s'il s'agit des enfants on sait que les enfants sont touchés autant que les adultes). En revanche cette variable pourra etre intéressante pour la comparer avec les résultats de tests sanguins
    - target / viral : les doubles maladies sont tres rares. Rhinovirus/Enterovirus positif - covid-19 négatif ? -> hypothese a tester ? mais il est possible que la région est subie une épidémie de ce virus. De plus on peut tres bien avoir 2 virus en meme temps. Tout ca n'a aucun lien avec le covid-19
    
    
    
## Analyse plus détaillée

- **Relation Variables / Variables** :
    - blood_data / blood_data : certaines variables sont tres corrélées : +0.9 (a suveiller plus tard)
    - blood_data / age : tres faible corrélation entre age et taux sanguins
    - viral / viral : influenza rapid test donne de mauvais résultats, il fauda peut-etre la laisser tomber
    - relation maladie / blood data : Les taux sanguins entre malades et covid-19 sont différents
    - relation hospitalisation / est malade : 
    - relation hospitalisation / blood : intéressant dans le cas ou on voudrait prédire dans quelle service un patient devrait aller


- **NaN analyse** : viral : 1350(92/8), blood : 600(87/13), both : 90

### hypotheses nulle (H0): 

- Les individus atteints du covid-19 ont des taux de Leukocytes, Monocytes, Platelets significativement différents
    - H0 = Les taux moyens sont ÉGAUX chez les individus positifs et négatifs

- Les individus atteints d'une quelconque maladie ont des taux significativement différents

### Analyse de la forme des données
"""

df = covid.copy()

df.shape

df.dtypes.value_counts().plot.pie()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

(df.isna().sum()/df.shape[0]).sort_values(ascending=True)

"""## Analyse du Fond
### 1. Visulation initiale - Elimination des colonnes inutiles
"""

df = df[df.columns[df.isna().sum()/df.shape[0] <0.9]]
df.head()

plt.figure(figsize=(20,10))
sns.heatmap(df.isna(), cbar=False)

# df = df.drop('Patient ID', axis=1)

df.head()

"""### Examen de la colonne target"""

df['SARS-Cov-2 exam result'].value_counts(normalize=True)

"""### histogrames des variables continues """

for col in df.select_dtypes('float'):
    plt.figure()
    sns.distplot(df[col])

sns.distplot(df['Patient age quantile'], bins=20)

df['Patient age quantile'].value_counts()

"""### Variables Qualitatives"""

for col in df.select_dtypes('object'):
    print(f'{col :-<50} {df[col].unique()}')

for col in df.select_dtypes('object'):
    plt.figure()
    df[col].value_counts().plot.pie()

"""## Relation Target / Variables
### Création de sous-ensembles positifs et négatifs
"""

positive_df = df[df['SARS-Cov-2 exam result'] == 'positive']

negative_df = df[df['SARS-Cov-2 exam result'] == 'negative']

"""### Création des ensembles Blood et viral"""

missing_rate = df.isna().sum()/df.shape[0]

blood_columns = df.columns[(missing_rate < 0.9) & (missing_rate >0.88)]

viral_columns = df.columns[(missing_rate < 0.88) & (missing_rate > 0.75)]

"""## Target / Blood"""

for col in blood_columns:
    plt.figure()
    sns.distplot(positive_df[col], label='positive')
    sns.distplot(negative_df[col], label='negative')
    plt.legend()

"""### Relation Target / age"""

sns.countplot(x='Patient age quantile', hue='SARS-Cov-2 exam result', data=df)

"""### Relation Target / Viral"""

pd.crosstab(df['SARS-Cov-2 exam result'], df['Influenza A'])

for col in viral_columns:
    plt.figure()
    sns.heatmap(pd.crosstab(df['SARS-Cov-2 exam result'], df[col]), annot=True, fmt='d')

"""## Analyse un peu plus Avancée
### Relation Variables / Variables
### relations Taux Sanguin
"""

sns.pairplot(df[blood_columns])

sns.clustermap(df[blood_columns].corr())

"""## Relation Age / Sang"""

for col in blood_columns:
    plt.figure()
    sns.lmplot(x='Patient age quantile', y=col, hue='SARS-Cov-2 exam result', data=df)



df.corr()['Patient age quantile'].sort_values()

"""### Relation entre Influenza et rapid test"""

pd.crosstab(df['Influenza A'], df['Influenza A, rapid test'])

pd.crosstab(df['Influenza B'], df['Influenza B, rapid test'])

"""### relation Viral / sanguin 
#### Création d'une nouvelle variable "est malade"
"""

df['est malade'] = np.sum(df[viral_columns[:-2]] == 'detected', axis=1) >=1

df.head()

malade_df = df[df['est malade'] == True]
non_malade_df = df[df['est malade'] == False]

for col in blood_columns:
    plt.figure()
    sns.distplot(malade_df[col], label='malade')
    sns.distplot(non_malade_df[col], label='non malade')
    plt.legend()

def hospitalisation(df):
    if df['Patient addmited to regular ward (1=yes, 0=no)'] == 1:
        return 'surveillance'
    elif df['Patient addmited to semi-intensive unit (1=yes, 0=no)'] == 1:
        return 'soins semi-intensives'
    elif df['Patient addmited to intensive care unit (1=yes, 0=no)'] == 1:
        return 'soins intensifs'
    else:
        return 'inconnu'

df['statut'] = df.apply(hospitalisation, axis=1)

df.head()

for col in blood_columns:
    plt.figure()
    for cat in df['statut'].unique():
        sns.distplot(df[df['statut']==cat][col], label=cat)
    plt.legend()

df[blood_columns].count()

df[viral_columns].count()

df1 = df[viral_columns[:-2]]
df1['covid'] = df['SARS-Cov-2 exam result']
df1.dropna()['covid'].value_counts(normalize=True)

df2 = df[blood_columns]
df2['covid'] = df['SARS-Cov-2 exam result']
df2.dropna()['covid'].value_counts(normalize=True)

"""## T-Test"""

from scipy.stats import ttest_ind

positive_df

balanced_neg = negative_df.sample(positive_df.shape[0])

def t_test(col):
    alpha = 0.02
    stat, p = ttest_ind(balanced_neg[col].dropna(), positive_df[col].dropna())
    if p < alpha:
        return 'H0 Rejetée'
    else :
        return 0

for col in blood_columns:
    print(f'{col :-<50} {t_test(col)}')









