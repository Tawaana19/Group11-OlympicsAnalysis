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

