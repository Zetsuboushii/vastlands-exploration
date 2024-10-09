import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import calculate_age, get_day_of_year


def create_gender_distribution(df_characters):
    sex_counts = df_characters['sex'].value_counts()
    labels = sex_counts.index.tolist()
    sizes = sex_counts.values.tolist()

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Male and Female Characters')
    plt.show()


def create_age_distribution_200y_focus(df_characters):
    df_characters['age'] = df_characters['birthday'].apply(calculate_age)
    df_age = df_characters.dropna(subset=['age'])
    df_age_under_200 = df_age[df_age['age'] <= 200]

    male_characters = df_age_under_200[df_age_under_200['sex'] == 'm']
    female_characters = df_age_under_200[df_age_under_200['sex'] == 'w']

    plt.figure(figsize=(8, 6))

    plt.hist(male_characters['age'], bins=20, edgecolor='black', alpha=0.5, label='Male',
             color='blue')
    plt.hist(female_characters['age'], bins=20, edgecolor='black', alpha=0.5, label='Female',
             color='pink')

    plt.xticks(range(0, 201, 10))
    plt.title('Distribution of Ages (Up to 200) by Gender')
    plt.xlabel('Age')
    plt.ylabel('Number of Characters')
    plt.legend(loc='upper right')
    plt.show()


def create_age_distribution_normalized(df_characters, df_races):
    df_characters['age'] = df_characters['birthday'].apply(calculate_age)
    df_characters = df_characters.dropna(subset=['age'])
    df_races['ageavg'] = pd.to_numeric(df_races['ageavg'], errors='coerce')
    df_races = df_races.dropna(subset=['ageavg'])
    df_races.loc[:, "norm_metric"] = df_races['ageavg'] / 100

    merge_df = pd.merge(df_characters, df_races, left_on="race", right_on="name", how="inner")
    merge_df["normed_age"] = merge_df["age"] / merge_df["norm_metric"]

    male_characters = merge_df[merge_df['sex'] == 'm']
    female_characters = merge_df[merge_df['sex'] == 'w']

    plt.figure(figsize=(8, 6))

    plt.hist(male_characters["normed_age"], bins=20, edgecolor='black', alpha=0.5, label='Male',
             color='blue')
    plt.hist(female_characters["normed_age"], bins=20, edgecolor='black', alpha=0.5, label='Female',
             color='pink')

    plt.xticks(range(0, 201, 10))
    plt.title('Distribution of Ages normalized on human age by gender')
    plt.xlabel('Age')
    plt.ylabel('Number of Characters')
    plt.legend(loc='upper right')
    plt.show()


def create_birthday_data_presence_pie_chart(df_characters: pd.DataFrame):
    birthday_counts = (df_characters["birthday"] != "").value_counts()
    labels = ["Given", "Not given"]
    sizes = [birthday_counts[True], birthday_counts[False]]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Characters with birthdays assigned or not")
    plt.show()


def create_birthday_distribution_clock_diagram(df_characters: pd.DataFrame):
    # Birthdays that are actually set
    characters_with_birthdays = df_characters[df_characters["birthday"] != ""]

    unique_races = df_characters["race"].unique()
    color_map = cm.get_cmap("Set2", len(unique_races))
    colors = {race: color_map(i) for i, race in enumerate(unique_races)}

    plt.figure(figsize=(9, 6))
    ax = plt.subplot(111, polar=True)

    grouping = characters_with_birthdays.groupby("race")
    for race, race_df in grouping:
        set_birthdays = race_df["birthday"]
        days_of_the_year = set_birthdays.apply(get_day_of_year)
        ax.scatter(days_of_the_year, np.ones(len(race_df)) * .97, label=race, color=colors[race],
                   linewidths=0.2)

    plt.setp(ax.get_yticklabels(), visible=False)
    ax.set_xticks(np.linspace(0, 2 * np.pi, 12, endpoint=False))
    ax.set_xticklabels(range(1, 13))
    ax.set_theta_direction(-1)
    ax.set_theta_offset(-np.pi / 2)

    ax.legend(loc="center right", bbox_to_anchor=(1.4, 0.5), title="Rassen")
    plt.ylim(0, 1)

    plt.title("Birthday distribution by race")
    plt.show()
