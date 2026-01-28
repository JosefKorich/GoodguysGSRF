import matplotlib.pyplot as plt

# Compute medal counts by country & sport
medal_events = athletes_df[athletes_df['Medal'].notna()].copy()
# Consider each (Year, Team, Sport, Event) where a medal was earned as one medal event
medal_events = medal_events.drop_duplicates(subset=['Year','Team','Sport','Event'])
country_sport_medals = medal_events.groupby(['Team','Sport']).size().reset_index(name='MedalCount')
total_medals_by_team = country_sport_medals.groupby('Team')['MedalCount'].sum().reset_index(name='TotalMedals')
country_sport_medals = pd.merge(country_sport_medals, total_medals_by_team, on='Team')
country_sport_medals['Importance'] = country_sport_medals['MedalCount'] / country_sport_medals['TotalMedals']

# Identify single-sport countries (medals all in one sport)
single_sport_teams = country_sport_medals.groupby('Team')['Sport'].nunique().reset_index()
single_sport_teams = single_sport_teams[single_sport_teams['Sport']==1]['Team'].tolist()
print(f"Countries that have medaled in only one sport: {single_sport_teams}")

# Example: visualize a specific country's medal distribution by sport using a radar chart
def plot_medal_distribution(country):
    df = country_sport_medals[country_sport_medals['Team']==country]
    if df.empty:
        print(f"No medals for country: {country}")
        return
    # Sports and their importance
    categories = df['Sport'].tolist()
    values = df['Importance'].tolist()
    # Close the radar chart loop
    categories += [categories[0]]
    values += [values[0]]
    angles = [n / float(len(categories)-1) * 2*np.pi for n in range(len(categories))]
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, polar=True)
    ax.plot(angles, values, linewidth=2, linestyle='solid')
    ax.fill(angles, values, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_yticklabels(['0%', '25%', '50%', '75%', '100%'])
    ax.set_title(f'{country} Medal Distribution by Sport')
    plt.tight_layout()
    plt.show()

# Plot an example: Kosovo (KOS) which is known for Judo medals only
plot_medal_distribution('Kosovo')
