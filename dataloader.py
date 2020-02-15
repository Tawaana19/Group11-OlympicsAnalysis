import pandas as pd

athlete_events = pd.read_csv('./data/athlete_events.csv')
country_dictionary = pd.read_csv("./data/country_dictionary.csv")
noc_regions = pd.read_csv("./data/noc_regions.csv")
summer = pd.read_csv("./data/summer.csv")
winter = pd.read_csv("./data/winter.csv")

print(athlete_events.head())