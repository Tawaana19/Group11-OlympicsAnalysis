def stacked_bar_plot():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import chart_studio
    import numpy as np
    import plotly.graph_objs as go
    preds = pd.read_csv('./dat/final_predictions.csv')
    preds = preds.drop(['Unnamed: 0'], axis=1)
    preds = preds.drop(['Host','Athletes','Events','Athletes per Event','NOC'], axis=1)
    top_10 = preds[:10]
    top_10 = top_10.pivot_table(index='Region')
    top_10 = top_10[['Golds Prediction','Silvers Prediction','Bronzes Prediction','Medals Prediction']]
    top_10.sort_values(by='Medals Prediction', ascending=False, inplace=True)
    top_10.drop(['Medals Prediction'], axis=1, inplace=True)
    colors = ["gold", "black","sienna"]
    ax = top_10.plot.bar(stacked=True, color=colors, figsize=(10,7),rot=0)
    ax.set_xlabel("")
    ax.set_ylabel("Medals")
    plt.show()

def medal_trends_1():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import chart_studio
    import numpy as np
    import plotly.graph_objs as go
    previous = pd.read_csv('./dat/previous_medals.csv')
    previous = previous.drop(['Unnamed: 0'], axis=1)

    previous = previous.loc[(previous['NOC'] == 'USA') | (previous['NOC'] == 'CHN') | 
                 (previous['NOC'] == 'GBR') | (previous['NOC'] == 'RUS') | 
                 (previous['NOC'] == 'GER')]

    previous = previous.drop(['Silvers','Bronzes','NOC','Summer'], axis=1)

    preds = pd.read_csv('./dat/final_predictions.csv')
    preds = preds.drop(['Unnamed: 0'], axis=1)
    preds = preds.drop(['Silvers Prediction','Bronzes Prediction','NOC','Host','Athletes','Events','Athletes per Event'], axis=1)
    preds = preds[:5]
    preds['Year'] = 2020
    preds = preds.rename(columns={'Golds Prediction': 'Golds','Medals Prediction': 'Medals' })

    final_preds = pd.concat([previous,preds])

    plt1 = final_preds.pivot(index='Year', columns='Region', values='Medals')
    plt1 = plt1[['USA','China','UK','Russia','Germany']]

    styles=['bo-', 'ro-', 'yo-','go-','ko-']
    ax = plt1.plot(style=styles,xticks=[2000,2004,2008,2012,2016,2020])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Total Medals")
    ax.legend(labels = ('United States','China','Great Britain','Russia','Germany'),
                loc='right',bbox_to_anchor=(1.4,0.5))
    plt.show()

def medal_trends_2():
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import chart_studio
    import numpy as np
    import plotly.graph_objs as go
    previous = pd.read_csv('./dat/previous_medals.csv')
    previous = previous.drop(['Unnamed: 0'], axis=1)

    previous = previous.loc[(previous['NOC'] == 'USA') | (previous['NOC'] == 'CHN') | 
                 (previous['NOC'] == 'GBR') | (previous['NOC'] == 'RUS') | 
                 (previous['NOC'] == 'GER')]

    previous = previous.drop(['Silvers','Bronzes','NOC','Summer'], axis=1)

    preds = pd.read_csv('./dat/final_predictions.csv')
    preds = preds.drop(['Unnamed: 0'], axis=1)
    preds = preds.drop(['Silvers Prediction','Bronzes Prediction','NOC','Host','Athletes','Events','Athletes per Event'], axis=1)
    preds = preds[:5]
    preds['Year'] = 2020
    preds = preds.rename(columns={'Golds Prediction': 'Golds','Medals Prediction': 'Medals' })

    final_preds = pd.concat([previous,preds])

    plt2 = final_preds.pivot(index='Year', columns='Region', values='Golds')
    plt2 = plt2[['USA','China','UK','Russia','Germany']]

    styles=['bo-', 'ro-', 'yo-','go-','ko-']
    ax = plt2.plot(style=styles,xticks=[2000,2004,2008,2012,2016,2020])
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Gold Medals")
    ax.legend(labels = ('United States','China','Great Britain','Russia','Germany'),
                loc='right',bbox_to_anchor=(1.4,0.5))
    plt.show()


def world_map():
    from IPython.core.display import display, HTML
    from IPython.display import FileLink
    local_file = FileLink('./dat/country_medals.html')
    display(local_file)
    # display(HTML('./dat/country_medals.html'))
    