def medal_vs_gdp_plot(medals, NOC_input_AE, NOC_input_GDP, medal_type):
    """
    plots for country & type of medal
    :param medals: medals grouped in DataFrame
    :param NOC_input: str (3 letter country code)
    :param medal_type: 'Golds', 'Silvers' 'Bronzes', 'Total'
    :return: None, plot shows
    """
    thisdict = {
        "gold": 0,
        "silver": 1,
        "bronze": 2,
        "total": 3
    }
    import numpy as np
    import math
    import matplotlib.patches as mpatches
    assert isinstance(NOC_input_AE, str)
    choice = thisdict[medal_type]
    medal_arr = []
    # Golds tuple (year, Gold, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Golds'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Silver tuple (year, Silvers, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Silvers'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Bronze tuple (year, Bronzes, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Bronzes'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # IND Medal tuple (year, Medals, count)

    temp = []
    for i in range(len(medal_arr[0])):
        try:
            temp.append((medal_arr[0][i][0], medal_arr[0][i][1] + medal_arr[1][i][1] + medal_arr[2][i][1]))
        except:
            temp.append((medal_arr[0][i][0], None))
    medal_arr.append(temp)

    gdp_list = list(it.chain(*gdp.loc[gdp['Country Code'] == NOC_input_GDP, '1960':'2018'].values.tolist()))[0::4]
    gdp_list = [None if math.isnan(v) else v for v in gdp_list]

    # Plotting of Medal Count vs GDP
    sns.set(style="white")
    plt.figure(figsize=(10, 8))
    x_medal = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    y_medal = np.array(list(i[1] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    medal_df = pd.DataFrame({"Year": x_medal.astype(int),
                             "Medal_count": y_medal.astype(int)
                             })

    ax1 = sns.lineplot(x="Year", y="Medal_count", data=medal_df, color='blue')
    plt.title(NOC_input_AE)

    ax2 = ax1.twinx()
    x_gdp = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(gdp_list).astype(np.double))]
    y_gdp = np.array(gdp_list)[np.isfinite(np.array(gdp_list).astype(np.double))]
    gdp_df = pd.DataFrame({"Year": x_gdp.astype(int),
                           "GDP": y_gdp.astype(int)
                           })
    sns.lineplot(x="Year", y="GDP", data=gdp_df, ax=ax2, color='red', legend='brief')
    red_patch = mpatches.Patch(color='red', label='GDP per Capita')
    blue_patch = mpatches.Patch(color='blue', label='Number of ' + medal_type + ' medals')
    plt.legend(handles=[red_patch, blue_patch])


def medal_vs_gdp_scatter(NOC_input_AE, NOC_input_GDP, medal_type, save) :
    """
    plots for country & type of medal
    :param medals: medals grouped in DataFrame
    :param NOC_input: str (3 letter country code)
    :param medal_type: 'Golds', 'Silvers' 'Bronzes', 'Total'
    :return: None, plot shows
    """
    thisdict = {
        "gold": 0,
        "silver": 1,
        "bronze": 2,
        "total": 3
    }

    #imports
    import numpy as np
    import math
    import matplotlib.patches as mpatches
    import pandas as pd
    import seaborn as sns ; sns.set()
    import matplotlib.pyplot as plt
    import itertools as it

    # Data set to work with
    athlete_events = pd.read_csv('./../dat/athlete_events_clean.csv')
    gdp = pd.read_csv('./../dat/WorldBank.csv')
    games = pd.read_pickle('./../dat/games.pkl')
    medals = pd.read_pickle('./../dat/medals.pkl')

    assert isinstance(NOC_input_AE, str)

    choice = thisdict[medal_type]
    medal_arr = []
    # Golds tuple (year, Gold, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Golds'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Silver tuple (year, Silvers, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Silvers'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Bronze tuple (year, Bronzes, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Bronzes'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Medal tuple (year, Medals, count)
    temp = []
    for i in range(len(medal_arr[0])):
        try:
            temp.append((medal_arr[0][i][0], medal_arr[0][i][1] + medal_arr[1][i][1] + medal_arr[2][i][1]))
        except:
            temp.append((medal_arr[0][i][0], None))
    medal_arr.append(temp)

    gdp_list = list(it.chain(*gdp.loc[gdp['Country Code'] == NOC_input_GDP, '1960':'2018'].values.tolist()))[0::4]
    gdp_list = [None if math.isnan(v) else v for v in gdp_list]

    # Plotting of Medal Count vs GDP
    sns.set(style="white")
    plt.figure(figsize=(10, 7))
    x_medal = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    y_medal = np.array(list(i[1] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    medal_df = pd.DataFrame({"Year": x_medal.astype(int),
                             "Medal_count": y_medal.astype(int)
                             })

    ax1 = sns.regplot(x="Year", y="Medal_count", data=medal_df, color='green')
    ax1.set(xlabel='Year', ylabel=medal_type.capitalize() + ' medal count')
    plt.title(NOC_input_AE)

    ax2 = ax1.twinx()
    x_gdp = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(gdp_list).astype(np.double))]
    y_gdp = np.array(gdp_list)[np.isfinite(np.array(gdp_list).astype(np.double))]
    gdp_df = pd.DataFrame({"Year": x_gdp.astype(int),
                           "GDP": y_gdp.astype(int)
                           })
    sns.regplot(x="Year", y="GDP", data=gdp_df, ax=ax2, color='red')
    ax2.set(ylabel='GDP per Capita ($)')
    red_patch = mpatches.Patch(color='red', label='GDP per Capita ($)')
    blue_patch = mpatches.Patch(color='green', label='Number of ' + medal_type + ' medals')
    plt.legend(handles=[red_patch, blue_patch], loc='upper left')

    if save:
        plt.savefig(NOC_input_AE + ".png")
    else:
        pass

    return plt


def medal_vs_gdp_EQ(medals, NOC_input_AE, NOC_input_GDP, medal_type):
    """
    plots for country & type of medal
    :param medals: medals grouped in DataFrame
    :param NOC_input: str (3 letter country code)
    :param medal_type: 'Golds', 'Silvers' 'Bronzes', 'Total'
    :return: None, plot shows
    """
    thisdict = {
        "gold": 0,
        "silver": 1,
        "bronze": 2,
        "total": 3
    }

    from sklearn import linear_model
    import numpy as np
    import math
    import matplotlib.patches as mpatches
    import pandas as pd
    import seaborn as sns ; sns.set()
    import matplotlib.pyplot as plt
    import itertools as it

    athlete_events = pd.read_csv('./../dat/athlete_events_clean.csv')
    gdp = pd.read_csv('./../dat/WorldBank.csv')
    games = pd.read_pickle('./../dat/games.pkl')
    medals = pd.read_pickle('./../dat/medals.pkl')

    assert isinstance(NOC_input_AE, str)
    choice = thisdict[medal_type]
    medal_arr = []
    # Golds tuple (year, Gold, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Golds'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Silver tuple (year, Silvers, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Silvers'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # Bronze tuple (year, Bronzes, count)
    temp = []
    for i in range(15):
        try:
            temp.append((1960 + i * 4,
                         medals.groupby(by=['Year', 'NOC']).get_group((1960 + i * 4, NOC_input_AE))['Bronzes'].sum()))
        except:
            temp.append((1960 + i * 4, None))
    medal_arr.append(temp)

    # IND Medal tuple (year, Medals, count)

    temp = []
    for i in range(len(medal_arr[0])):
        try:
            temp.append((medal_arr[0][i][0], medal_arr[0][i][1] + medal_arr[1][i][1] + medal_arr[2][i][1]))
        except:
            temp.append((medal_arr[0][i][0], None))
    medal_arr.append(temp)

    gdp_list = list(it.chain(*gdp.loc[gdp['Country Code'] == NOC_input_GDP, '1960':'2018'].values.tolist()))[0::4]
    gdp_list = [None if math.isnan(v) else v for v in gdp_list]

    # Plotting of Medal Count vs GDP
    #     sns.set(style="white")
    #     plt.figure(figsize=(10,8))
    x_medal = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    y_medal = np.array(list(i[1] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(list(i[1] if i is not None else i for i in medal_arr[choice])).astype(np.double))]
    medal_df = pd.DataFrame({"Year": x_medal.astype(int),
                             "Medal_count": y_medal.astype(int)
                             })
    regr1 = linear_model.LinearRegression()
    X = x_medal.reshape(-1, 1)
    Y = y_medal.reshape(-1, 1)
    regr1.fit(X, Y)

    x_gdp = np.array(list(i[0] if i is not None else i for i in medal_arr[choice]))[
        np.isfinite(np.array(gdp_list).astype(np.double))]
    y_gdp = np.array(gdp_list)[np.isfinite(np.array(gdp_list).astype(np.double))]
    regr2 = linear_model.LinearRegression()
    X = x_gdp.reshape(-1, 1)
    Y = y_gdp.reshape(-1, 1)
    regr2.fit(X, Y)


    return float(regr1.coef_[0]), float(regr2.coef_[0])

def plot_rate():
    """
    Helper function to Runs the Scatter iteratively for the plot
    """
    import matplotlib.patches as mpatches
    import pandas as pd
    import seaborn as sns ; sns.set()
    import matplotlib.pyplot as plt
    import itertools as it

    # Data set to work with
    medals = pd.read_pickle('./data/medals.pkl')

    # iterate through several NOC indexes
    noc_test_developed = ['USA', 'CAN', 'ITA', 'JPN']
    developed = []
    for i in noc_test_developed:
        developed.append((i, medal_vs_gdp_EQ(medals, i, i, 'total')[0], medal_vs_gdp_EQ(medals, i, i, 'total')[1]))

    noc_test_developing = ['JAM', 'KEN', 'BRA', 'IND']
    developing = []
    for i in noc_test_developing:
        #     print(i+ " medal ROC: "+ str(medal_vs_gdp_plot_EQ(medals,i,i,'total')[0]) + ", GDP ROC: " + str(medal_vs_gdp_plot_EQ(medals,i,i,'total')[1]))
        developing.append((i, medal_vs_gdp_EQ(medals, i, i, 'total')[0], medal_vs_gdp_EQ(medals, i, i, 'total')[1]))


    # Make pandas dataframes to plot
    roc_data = pd.DataFrame({"NOC": noc_test_developed,
                             "Medal_coeff": list(i[1] for i in developed),
                             "GDP_coeff": list(i[2] for i in developed),
                             })
    roc_data1 = pd.DataFrame({"NOC": noc_test_developing,
                              "Medal_coeff": list(i[1] for i in developing),
                              "GDP_coeff": list(i[2] for i in developing),
                              })

    # Configure Plots
    plt.figure(figsize=(10, 7))
    sns.set(style="ticks")
    p1 = sns.scatterplot(x="GDP_coeff", y="Medal_coeff", data=roc_data, color='red')
    p2 = sns.scatterplot(x="GDP_coeff", y="Medal_coeff", data=roc_data1, color='blue')

    for line in range(0, roc_data.shape[0]):
        p1.text(roc_data.GDP_coeff[line] - 30, roc_data.Medal_coeff[line] + 0.006,
                roc_data.NOC[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold', fontsize=15)

    for line in range(0, roc_data1.shape[0]):
        p1.text(roc_data1.GDP_coeff[line] - 30, roc_data1.Medal_coeff[line] + 0.006,
                roc_data1.NOC[line], horizontalalignment='left',
                size='medium', color='black', weight='semibold', fontsize=15)

    red_patch = mpatches.Patch(color='red', label='Developed')
    blue_patch = mpatches.Patch(color='blue', label='Developing')
    plt.legend(handles=[red_patch, blue_patch], loc='upper center')
    p1.set(xlabel='Rate of Change of GDP per Capita ($)', ylabel='Rate of Change of Medal Count')
    return plt
