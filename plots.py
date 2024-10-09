import matplotlib.pyplot as plt
import pandas as pd
from utils import calculate_age

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

    plt.hist(male_characters['age'], bins=20, edgecolor='black', alpha=0.5, label='Male', color='blue')
    plt.hist(female_characters['age'], bins=20, edgecolor='black', alpha=0.5, label='Female', color='pink')

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

    plt.hist(male_characters["normed_age"], bins=20, edgecolor='black', alpha=0.5, label='Male', color='blue')
    plt.hist(female_characters["normed_age"], bins=20, edgecolor='black', alpha=0.5, label='Female', color='pink')

    plt.xticks(range(0, 201, 10))
    plt.title('Distribution of Ages normalized on human age by gender')
    plt.xlabel('Age')
    plt.ylabel('Number of Characters')
    plt.legend(loc='upper right')
    plt.show()
