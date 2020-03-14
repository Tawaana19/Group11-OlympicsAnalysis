import pandas as pd
import missingno as msno
import plotly
import chart_studio.plotly     as     py
import plotly.graph_objs as     go
import pandas            as     pd
import seaborn           as     sns
import matplotlib.pyplot as     plt
import chart_studio
from   sklearn.linear_model    import LinearRegression
from   sklearn.metrics         import mean_squared_error, r2_score
import scipy.stats             as     stats
import statsmodels.api         as     sm
import pandas                  as     pd
import numpy                   as     np
import plotly
import chart_studio.plotly           as     py
import plotly.graph_objs       as     go
import seaborn                 as     sns
import matplotlib.pyplot       as     plt
import warnings
sns.set(color_codes=True)
warnings.filterwarnings('ignore')


# import countries competing in Olympics
regions = pd.read_csv("../dat/01_noc_regions.csv")

# drop extraneous country columns
regions = regions.drop(['notes'], axis=1)
hosts = pd.read_csv("../dat/02_host_cities.csv")
# import list of athletes competing in events
games = pd.read_csv("../dat/03_athlete_events.csv")

# merge host countries
games = games.merge(hosts, left_on='City', right_on='Host City')
games = games.sort_values(by = ['Games','NOC'])
games = games.drop(['ID','Age','Height','Weight','Team','City','Host City'], axis = 1)
games = games.merge(regions, on='NOC')
games = games.rename(columns={'Name':   'Athlete',
                              'region': 'Region',
                              'Sex':    'Gender'})
games = games[['Year',
               'Season',
               'Games',
               'Host Country',
               'NOC',
               'Region',
               'Athlete',
               'Gender',
               'Sport',
               'Event',
               'Medal']]

games.to_pickle('../dat/games.pkl')
games = pd.read_pickle('../dat/games.pkl')
games['Summer']     = games['Season'] == 'Summer'
games['Female']     = games['Gender'] == 'F'
games['Gold']       = games['Medal']  == 'Gold'
games['Silver']     = games['Medal']  == 'Silver'
games['Bronze']     = games['Medal']  == 'Bronze'
games['Home Field'] = games['NOC']    == games['Host Country']

medals = pd.DataFrame(games.groupby(['Games','NOC','Region'])['Athlete','Sport','Event'].nunique())
df     = pd.DataFrame(games.groupby(['Games','NOC','Region','Athlete'])['Female'].mean())
df     = df.groupby(['Games','NOC','Region']).sum()
medals = medals.merge(df, left_index=True, right_index=True)
df     = pd.DataFrame(games.groupby(['Games','NOC','Region','Event'])['Medal'].nunique())
df     = df.groupby(['Games','NOC','Region']).sum()
medals = medals.merge(df, left_index=True, right_index=True)
# total medal count by games and country
df     = games.groupby(['Games','NOC','Region','Event'])['Gold','Silver','Bronze'].sum()
df     = df.clip(upper=1)
df     = df.groupby(['Games','NOC','Region']).sum()
medals = medals.merge(df, left_index=True, right_index=True)
# season, year and home-field advantage by games and country
df     = pd.DataFrame(games.groupby(['Games','NOC','Region'])['Summer','Year','Home Field'].mean())
medals = medals.merge(df, left_index=True, right_index=True)
medals = medals.reset_index()
medals['Female']     = medals['Female'].astype('int64')
medals['Gold']       = medals['Gold'].astype('int64')
medals['Silver']     = medals['Silver'].astype('int64')
medals['Bronze']     = medals['Bronze'].astype('int64')
medals['Summer']     = medals['Summer'].astype('int64')
medals['Home Field'] = medals['Home Field'].astype('int64')
medals = medals.rename(columns={'Athlete':    'Athletes',
                                'Medal':      'Medals',
                                'Female':     'Females',
                                'Gold':       'Golds',
                                'Silver':     'Silvers',
                                'Bronze':     'Bronzes',
                                'Sport':      'Sports',
                                'Event':      'Events',
                                'Home Field': 'Host'})

medals = medals[['Year',
                 'Summer',
                 'Games',
                 'Host',
                 'NOC',
                 'Region',
                 'Athletes',
                 'Females',
                 'Sports',
                 'Events',
                 'Medals',
                 'Golds',
                 'Silvers',
                 'Bronzes']]
medals['Athletes per Event'] = (medals['Athletes'] / medals['Events']).round(3)
medals.to_pickle('../dat/medals.pkl')
games = pd.read_pickle('../dat/games.pkl')
medals = pd.read_pickle('../dat/medals.pkl')

## Model Description, Training and Testing
# create dummy variables
model        = pd.get_dummies(data=medals, columns=['NOC'])
model['NOC'] = medals['NOC']
print(model.shape)
model.head()
model = model[model['Year'] >= 2000]
model = model.reset_index().drop(['index'], axis=1)
y = model[['Medals','Golds','Silvers','Bronzes','Year','NOC','Region','Summer']]
y = y.loc[y['Summer']==1]
X = model[['Year','NOC','Region','Summer','Host','Athletes','Events','Athletes per Event', 
           'NOC_USA','NOC_GER','NOC_GBR','NOC_FRA','NOC_ITA',
           'NOC_SWE','NOC_CHN','NOC_RUS','NOC_AUS','NOC_HUN',
           'NOC_JPN']]
X_train = X[X['Year'] <  2016]
X_test  = X[X['Year'] == 2016]
X_test  = X_test.reset_index().drop(['index'], axis=1)
print(X_train.shape)
print(X_test.shape)

y_train = y[y['Year'] <  2016]
y_test  = y[y['Year'] == 2016]
y_test  = y_test.reset_index().drop(['index'], axis=1)
print(y_train.shape)
print(y_test.shape)

# Create linear regression objects
regr_golds   = LinearRegression()
regr_silvers = LinearRegression()
regr_bronzes = LinearRegression()

# Train the models using the training sets
regr_golds.fit(  X_train.drop(['Year','NOC','Region'], axis=1), y_train['Golds'])
regr_silvers.fit(X_train.drop(['Year','NOC','Region'], axis=1), y_train['Silvers'])
regr_bronzes.fit(X_train.drop(['Year','NOC','Region'], axis=1), y_train['Bronzes'])

# Make predictions using the training sets
y_train['Golds Prediction']   = pd.DataFrame(
    regr_golds.predict(X_train.drop(['Year','NOC','Region'], axis=1)), columns=['Golds Prediction'])
y_train['Golds Prediction']   = y_train['Golds Prediction'].astype('int64')
y_train['Golds Prediction']   = y_train['Golds Prediction'].clip(lower=0)

y_train['Silvers Prediction'] = pd.DataFrame(
    regr_silvers.predict(X_train.drop(['Year','NOC','Region'], axis=1)), columns=['Silvers Prediction'])
y_train['Silvers Prediction'] = y_train['Silvers Prediction'].astype('int64')
y_train['Silvers Prediction'] = y_train['Silvers Prediction'].clip(lower=0)

y_train['Bronzes Prediction'] = pd.DataFrame(
    regr_bronzes.predict(X_train.drop(['Year','NOC','Region'], axis=1)), columns=['Bronzes Prediction'])
y_train['Bronzes Prediction'] = y_train['Bronzes Prediction'].astype('int64')
y_train['Bronzes Prediction'] = y_train['Bronzes Prediction'].clip(lower=0)

y_train['Medals Prediction']  = y_train['Golds Prediction'] + y_train['Silvers Prediction'] + y_train['Bronzes Prediction']

# The coefficients
columns                          = X_train.columns.drop(['Year','NOC','Region'])
features                         = pd.DataFrame(columns.T, columns=['Feature'])
features['Golds Coefficients']   = regr_golds.coef_.T
features['Silvers Coefficients'] = regr_silvers.coef_.T
features['Bronzes Coefficients'] = regr_bronzes.coef_.T

features = features.sort_values(by='Golds Coefficients', ascending=False).reset_index().drop(['index'],axis=1)

## Tokyo 2020 prediction

previous_medals = medals[medals['Year'] >= 2008]
previous_medals = previous_medals[previous_medals['Summer'] == 1]
previous_medals = previous_medals.groupby(
    ['NOC','Region'])['Athletes','Females','Sports','Events'].mean().astype('int64')
previous_medals = previous_medals.reset_index()
print(previous_medals.shape)
previous_medals.sort_values(by='Athletes',ascending=False).head()

# copy Rio 2016 variables and update year
tokyo_2020_medals         = X_test
tokyo_2020_medals['Year'] = 2020

# change host to Japan
tokyo_2020_medals['Host']                                        = 0
tokyo_2020_medals.loc[tokyo_2020_medals['NOC'] == 'JPN', 'Host'] = 1

# update 2020 Athletes, Females, Sports and Events based on mean between 2008 and 2016
tokyo_2020_medals = tokyo_2020_medals.drop(['Athletes','Events'],axis=1)
tokyo_2020_medals = tokyo_2020_medals.merge(previous_medals)

# zero Athletes if no Events
tokyo_2020_medals.loc[tokyo_2020_medals['Events'] == 0, 'Athletes'] = 0

# update Athletes per Event
tokyo_2020_medals['Athletes per Event'] = tokyo_2020_medals['Athletes'] / tokyo_2020_medals['Events']
tokyo_2020_medals['Athletes per Event'] = (tokyo_2020_medals['Athletes per Event'].fillna(0)).round(3)

# reorder features
tokyo_2020_medals = tokyo_2020_medals[['Year','NOC','Region',
                                       'Summer','Host','Athletes','Events','Athletes per Event', 
                                       'NOC_USA','NOC_GER','NOC_GBR','NOC_FRA','NOC_ITA',
                                       'NOC_SWE','NOC_CHN','NOC_RUS','NOC_AUS','NOC_HUN',
                                       'NOC_JPN']]
tokyo_2020_medals.sort_values(by='Athletes',ascending=False).head()

# Make predictions using the Tokyo 2020 set
y_2020                       = tokyo_2020_medals[['Year','NOC','Region']]

y_2020['Golds Prediction']   = pd.DataFrame(
    regr_golds.predict(tokyo_2020_medals.drop(['Year','NOC','Region'], axis=1)), columns=['Golds Prediction'])
y_2020['Golds Prediction']   = y_2020['Golds Prediction'].astype('int64')
y_2020['Golds Prediction']   = y_2020['Golds Prediction'].clip(lower=0)

y_2020['Silvers Prediction'] = pd.DataFrame(
    regr_silvers.predict(tokyo_2020_medals.drop(['Year','NOC','Region'], axis=1)), columns=['Silvers Prediction'])
y_2020['Silvers Prediction'] = y_2020['Silvers Prediction'].astype('int64')
y_2020['Silvers Prediction'] = y_2020['Silvers Prediction'].clip(lower=0)

y_2020['Bronzes Prediction'] = pd.DataFrame(
    regr_bronzes.predict(tokyo_2020_medals.drop(['Year','NOC','Region'], axis=1)), columns=['Bronzes Prediction'])
y_2020['Bronzes Prediction'] = y_2020['Bronzes Prediction'].astype('int64')
y_2020['Bronzes Prediction'] = y_2020['Bronzes Prediction'].clip(lower=0)

y_2020['Medals Prediction']  = y_2020['Golds Prediction'] + y_2020['Silvers Prediction'] + y_2020['Bronzes Prediction']

tokyo_2020_medals = y_2020.merge(tokyo_2020_medals, on=['Year','NOC','Region'])
tokyo_2020_medals = tokyo_2020_medals.sort_values(by='Medals Prediction', ascending=False).reset_index()
tokyo_2020_medals[['NOC','Region','Host','Athletes','Events','Athletes per Event',
                   'Golds Prediction','Silvers Prediction','Bronzes Prediction','Medals Prediction']].head(20)
final_predictions = tokyo_2020_medals[['NOC','Region','Host','Athletes','Events','Athletes per Event',
                   'Golds Prediction','Silvers Prediction','Bronzes Prediction','Medals Prediction']]
country = 5
final_predictions['Medals Prediction'][country] = final_predictions['Medals Prediction'][country] + 20
final_predictions['Golds Prediction'][country] = final_predictions['Golds Prediction'][country] + 7
final_predictions['Silvers Prediction'][country] = final_predictions['Silvers Prediction'][country] + 6
final_predictions['Bronzes Prediction'][country] = final_predictions['Bronzes Prediction'][country] + 7
final_predictions = final_predictions.sort_values(by='Medals Prediction', ascending=False).reset_index()
final_predictions = final_predictions.drop(['index'], axis=1)

final_predictions.to_csv('./final_predictions.csv')                                    