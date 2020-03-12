import pandas as pd
import seaborn as sns
from matplotlib.pylab import plt

athlete_events = pd.read_csv('./data/athlete_events.csv')
female_athletes = athlete_events.loc[athlete_events['Sex'] == 'F']
male_athletes = athlete_events.loc[athlete_events['Sex'] == 'M']
medal_winners = pd.read_csv("./data/summer_clean.csv")


def plot_us_women_win_ratio(medal_winners):
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


def plot_sex_ratio_year_wise(female_athletes, male_athletes):
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


def plot_country_sex_ratio(female_athletes, male_athletes,country):
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


plot_us_women_win_ratio(medal_winners)

plot_sex_ratio_year_wise(female_athletes, male_athletes)

plot_country_sex_ratio(female_athletes, male_athletes, "KSA")
