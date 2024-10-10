from typing import Optional

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


def create_weakness_distribution_pie_chart(df_enemies: pd.DataFrame):
    df_normalized = df_enemies.explode("weaknesses").reset_index(drop=True)
    weakness_group = df_normalized.groupby("weaknesses").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(weakness_group, labels=weakness_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Weaknesses')
    plt.show()


def create_resistance_distribution_pie_chart(df_enemies: pd.DataFrame):
    df_normalized = df_enemies.explode("resistances").reset_index(drop=True)
    resistances_group = df_normalized.groupby("resistances").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(resistances_group, labels=resistances_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Resistances')
    plt.tight_layout()
    plt.show()


def create_immunities_distribution_pie_chart(df_enemies: pd.DataFrame):
    df_normalized = df_enemies.explode("immunities").reset_index(drop=True)
    immunities_group = df_normalized.groupby("immunities").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(immunities_group, labels=immunities_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Immunities')
    plt.tight_layout()
    plt.show()


def create_combined_pie_charts(df_enemies: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Weaknesses Pie Chart
    df_normalized_weaknesses = df_enemies.explode("weaknesses").reset_index(drop=True)
    weakness_group = df_normalized_weaknesses.groupby("weaknesses").size().sort_values()
    axes[0].pie(weakness_group, labels=weakness_group.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Distribution of Weaknesses')

    # Resistances Pie Chart
    df_normalized_resistances = df_enemies.explode("resistances").reset_index(drop=True)
    resistances_group = df_normalized_resistances.groupby("resistances").size().sort_values()
    axes[1].pie(resistances_group, labels=resistances_group.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Distribution of Resistances')

    # Immunities Pie Chart
    df_normalized_immunities = df_enemies.explode("immunities").reset_index(drop=True)
    immunities_group = df_normalized_immunities.groupby("immunities").size().sort_values()
    axes[2].pie(immunities_group, labels=immunities_group.index, autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Distribution of Immunities')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def create_ability_score_distribution_plot(df_enemies: pd.DataFrame):
    as_avg = df_enemies[["str", "dex", "con", "int", "wis", "cha"]].mean()
    df_enemies['sum_stats'] = df_enemies[['str', 'dex', 'con', 'int', 'wis', 'cha']].sum(axis=1)
    mean_sum_stats = df_enemies['sum_stats'].mean().round(2)
    as_avg = as_avg.round(2)
    overall_avg = as_avg.mean().round(2)
    as_avg["overall_avg"] = overall_avg
    as_avg["as_avg"] = mean_sum_stats
    plt.figure(figsize=(8, 6))
    as_avg.plot(kind='bar')
    plt.title('Average Ability Score of Enemies', fontsize=16)
    plt.xlabel('Ability Score', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()


def create_stats_distribution_plot(df_enemies: pd.DataFrame):
    # TODO: add movement range, but this can be complicated because enemies can have multiple ways of moving ag. flying, swimming
    df_enemies["hp_ac_ratio"] = df_enemies["hp"] / df_enemies["ac"]
    stats_avg = df_enemies[["hp", "ac", "hp_ac_ratio"]].mean()
    stats_avg.round(2)
    stats_avg.plot(kind='bar')
    plt.title('Average Stats of Enemies', fontsize=16)
    plt.xlabel('Stats', fontsize=14)
    plt.ylabel('Average Value', fontsize=14)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    max_value = stats_avg.max()
    ticks = np.around(np.linspace(0, max_value, num=12))
    plt.yticks(ticks)
    plt.axhline(0, color='black', linewidth=1)
    plt.tight_layout()
    plt.show()

def create_character_class_bar_chart(df_characters: pd.DataFrame):
    character_classes = df_characters["character_class"].unique()
    sexes = df_characters["sex"].unique()

    plt.figure(figsize=(15, 8))

    n_classes = len(character_classes)
    n_sexes = len(sexes)
    group_width = 0.8
    bar_width = group_width / n_sexes

    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_sexes))

    for i, sex in enumerate(sexes):
        sex_data = df_characters[df_characters["sex"] == sex]
        class_counts = sex_data["character_class"].value_counts()
        x = np.arange(n_classes) + (i - (n_sexes - 1) / 2) * bar_width

        plt.bar(x, [class_counts.get(c, 0) for c in character_classes],
                width=bar_width, alpha=0.7, color=colors[i],
                label=f'Sex: {sex}', edgecolor='black')

    plt.xlabel('Character Class')
    plt.ylabel('Count')
    plt.title('Character Class Distribution by Sex')
    plt.xticks(np.arange(n_classes), character_classes, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def create_grouping_pie_chart(df: pd.DataFrame, group_column: str, title: str, legend: bool = True, legend_title: Optional[str] = None, min_percentage = .01):
    value_counts = df[group_column].value_counts()
    value_counts_sorted = value_counts.sort_values(ascending=False)

    total = value_counts_sorted.sum()
    percentages = value_counts_sorted / total

    main_slices = percentages[percentages >= min_percentage]
    small_slices = percentages[percentages < min_percentage]

    if not small_slices.empty:
        main_slices['Other'] = small_slices.sum()

    labels = main_slices.index.tolist()
    values = main_slices.values.tolist()

    base_colors = plt.cm.Set3.colors
    colors = [base_colors[i % len(base_colors)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = ax.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                      pctdistance=0.85, startangle=90)

    plt.setp(autotexts, size=8, weight="bold")
    plt.setp(texts, size=10)
    centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    fig.gca().add_artist(centre_circle)

    ax.set_title(title)
    if legend:
        ax.legend(wedges, labels,
                  title=legend_title,
                  loc="center right"
        )

    plt.tight_layout()
    plt.show()


'''
WIP Danger Level Calculation
- armor/ac value
- damage per turn capability
- action variety 
- status effects usable and weighting of their strength
- movement range and type with weighting
'''
