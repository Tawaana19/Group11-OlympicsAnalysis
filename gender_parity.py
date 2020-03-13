import pandas as pd
import seaborn as sns
from matplotlib.pylab import plt
from pygal_maps_world.maps import World

athlete_events = pd.read_csv('./data/athlete_events.csv')
female_athletes = athlete_events.loc[athlete_events['Sex'] == 'F']
male_athletes = athlete_events.loc[athlete_events['Sex'] == 'M']
medal_winners = pd.read_csv("./data/summer_clean.csv")
country_ratios = female_athletes[["NOC","Sex"]].groupby(["NOC"]).count()/(female_athletes[["NOC","Sex"]].groupby(["NOC"]).count()+male_athletes[["NOC","Sex"]].groupby(["NOC"]).count())


def plot_us_women_win_ratio():
    """This function plots the win ratio of US women over 120 years of Olympics
    :param medal_winners: dataframe object"""
    assert isinstance(medal_winners, pd.DataFrame),               "medal_winners must be of type pd.DataFrame"

    female_winners = medal_winners.loc[medal_winners['Gender'] == 'Women']
    male_winners = medal_winners.loc[medal_winners['Gender'] == 'Men']
    female_winners_usa = female_winners.loc[female_winners['Country'] == 'USA'].groupby("Year").count()
    male_winners_usa = male_winners.loc[male_winners['Country'] == 'USA'].groupby("Year").count()
    win_ratio_usa = female_winners_usa/(female_winners_usa + male_winners_usa)
    win_ratio_usa = win_ratio_usa.fillna(0)
    win_ratio_usa = win_ratio_usa.reset_index()

    ax = sns.lineplot(x="Year", y="Medal", markers=True, color="red", dashes=False,
                  data=win_ratio_usa)
    ax.set(xlabel='Year', ylabel='Medal Win Ratio')
    plt.show()


def plot_sex_ratio_year_wise():
    """
    This function plots the female to male participation ratio summed for all countries year wise
    :param female_athletes: dataframe object
    :param male_athletes: dataframe object
    """

    assert isinstance(female_athletes, pd.DataFrame),             "female_athletes must be of type pd.DataFrame"
    assert isinstance(male_athletes, pd.DataFrame),               "male_athletes must be of type pd.DataFrame"

    female_count_year_wise = female_athletes[["NOC", "Sex", "Year"]].groupby("Year").count()
    male_count_year_wise = male_athletes[["NOC", "Sex", "Year"]].groupby("Year").count()

    sex_ratio_year_wise = female_count_year_wise["Sex"]/(female_count_year_wise["Sex"] + male_count_year_wise["Sex"])
    sex_ratio_year_wise = sex_ratio_year_wise.reset_index()
    sex_ratio_year_wise = sex_ratio_year_wise.fillna(0)

    ax = sns.lineplot(x="Year", y="Sex", markers=True, dashes=False, data=sex_ratio_year_wise)
    ax.set(xlabel='Year', ylabel='Sex Ratio')
    plt.show()


def plot_country_sex_ratio(country):
    """
    This function plots the sex ratio of input country over the years
    :param female_athletes: dataframe object
    :param male_athletes: dataframe object
    :param country: Country name whose sex ratio we want to plot
    """
    assert isinstance(female_athletes, pd.DataFrame),             "female_athletes must be of type pd.DataFrame"
    assert isinstance(male_athletes, pd.DataFrame),               "male_athletes must be of type pd.DataFrame"
    assert isinstance(country, str),                              "Country name must be an integer"

    country_ratios_year = female_athletes[["NOC", "Sex", "Year"]].groupby(["NOC", "Year"]).count() / \
                      (female_athletes[["NOC", "Sex", "Year"]].groupby(["NOC", "Year"]).count() +
                       male_athletes[["NOC", "Sex", "Year"]].groupby(["NOC", "Year"]).count())

    country_ratios_year = country_ratios_year.reset_index()

    country_plot = country_ratios_year.loc[country_ratios_year["NOC"] == country]
    country_plot = country_plot.fillna(0)

    ax = sns.lineplot(x="Year", y="Sex", markers=True, dashes=False, data=country_plot)
    ax.set(xlabel='Year', ylabel='Sex Ratio')
    plt.show()

def plot_global_sex_ratio():
    below_20 = country_ratios.loc[country_ratios["Sex"]<=0.20]
    below_20=below_20.reset_index()
    below_40 = country_ratios.loc[country_ratios["Sex"]<=0.40].loc[country_ratios["Sex"]>0.20]
    below_40=below_40.reset_index()
    below_60 = country_ratios.loc[country_ratios["Sex"]>0.40].loc[country_ratios["Sex"]<=0.60]
    below_60=below_60.reset_index()
    below_80 = country_ratios.loc[country_ratios["Sex"]>0.60].loc[country_ratios["Sex"]<=0.80]
    below_80=below_80.reset_index()
    above_80 = country_ratios.loc[country_ratios["Sex"]>0.80]
    above_80=above_80.reset_index()
    worldmap_chart = World()
    worldmap_chart.title = 'Countrywise Particpation Percentages of Females'
    worldmap_chart.add("0%-20%", ['af', 'dz', 'au', 'ar', 'am', 'be', 'bj', 'bz', 'cz', 'bw', 'bn', 'cl', 'dk',
                                    'dj', 'eg', 'er', 'fi', 'gh', 'ht', 'in', 'ir', 'iq', 'sa', 'kw', 'ly', 'lr',
                                    'lb', 'lu', 'my', 'ma', 'mw', 'mc', 'mr', 'ni', 'om', 'pk', 'py', 'ph', 'pt',
                                    'pr', 'zw', 'de', 'rs', 'sh', 'sm', 'so', 'sd', 'ch', 'sz', 'sy', 'tz', 'tg',
                                    'tn', 'tr', 'ae', 'ug', 'uy', 'vn', 'ye', 'mk', 'zm'])
    worldmap_chart.add('20%-40%', ['al', 'ad', 'au', 'at', 'az', 'bi', 'bo', 'br', 'bh', 'bg', 'bf', 'cf', 'kh', 'ca', 'cg', 
    'td', 'cm', 'cd', 'co', 'cr', 'hr', 'cu', 'cy', 'cz',  'do', 'ec', 'sv', 'es', 'ee', 'et', 'ru',
    'fr', 'de', 'ga', 'gm', 'gb', 'gw', 'ge', 'gq', 'gr', 'gn', 'gu', 'gy', 'cn', 'hn', 'hu', 'id',
    'ie', 'is', 'il', 'it', 'jo', 'jp', 'kz', 'ke', 'kg',  'kr', 'la', 'lv', 'ls', 'li', 'lt', 'md', 'mv', 'mx', 'mn', 
    'mk', 'ml', 'mt', 'me', 'mz', 'mu', 'mm', 'na', 'nl', 'np', 'ng', 'ne', 'no', 'nz', 'pa', 'pe', 'ps', 'pg', 'pl', 'ro', 
    'za', 'rw', 'sc', 'sg', 'sl', 'si', 'rs', 'lk', 'sd', 'st', 'sr', 'sk', 'se', 'cz', 'th', 'tj', 'tw',  
    'ru', 'us', 'uz', 've', 'zw'])
    worldmap_chart.add('40%-60%', ['ao', 'bt', 'by', 'cn', 'cv', 'jm', 'mg', 'kp', 'ru', 'tm', 'tl', 'ua', 'vn'])
    worldmap_chart.add('60%-80%',[])
    worldmap_chart.add('80%-100%',[])
    worldmap_chart.render_in_browser() 
