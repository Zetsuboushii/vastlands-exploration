import ast
from typing import Optional

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from adjustText import adjust_text
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from decorators import include_plot
from utils import calculate_age, get_day_of_year
import matplotlib.lines as mlines
import os


def create_gender_distribution(characters, **kwargs):
    sex_counts = characters['sex'].value_counts()
    labels = sex_counts.index.tolist()
    sizes = sex_counts.values.tolist()

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Male and Female Characters')
    plt.show()


def create_age_distribution_200y_focus(characters, **kwargs):
    characters['age'] = characters['birthday'].apply(calculate_age)
    df_age = characters.dropna(subset=['age'])
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


def create_age_distribution_normalized(characters, races, **kwargs):
    characters['age'] = characters['birthday'].apply(calculate_age)
    characters = characters.dropna(subset=['age'])
    races['ageavg'] = pd.to_numeric(races['ageavg'], errors='coerce')
    races = races.dropna(subset=['ageavg'])
    races.loc[:, "norm_metric"] = races['ageavg'] / 100

    merge_df = pd.merge(characters, races, left_on="race", right_on="name", how="inner")
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


def create_birthday_data_presence_pie_chart(characters: pd.DataFrame, **kwargs):
    birthday_counts = (characters["birthday"] != "").value_counts()
    labels = ["Given", "Not given"]
    sizes = [birthday_counts[True], birthday_counts[False]]

    plt.figure(figsize=(6, 6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Characters with birthdays assigned or not")
    plt.show()


def create_birthday_distribution_clock_diagram(characters: pd.DataFrame, **kwargs):
    # Birthdays that are actually set
    characters_with_birthdays = characters[characters["birthday"] != ""]

    unique_races = characters["race"].unique()
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


def create_weakness_distribution_pie_chart(enemies: pd.DataFrame, **kwargs):
    df_normalized = enemies.explode("weaknesses").reset_index(drop=True)
    weakness_group = df_normalized.groupby("weaknesses").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(weakness_group, labels=weakness_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Weaknesses')
    plt.show()


def create_resistance_distribution_pie_chart(enemies: pd.DataFrame, **kwargs):
    df_normalized = enemies.explode("resistances").reset_index(drop=True)
    resistances_group = df_normalized.groupby("resistances").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(resistances_group, labels=resistances_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Resistances')
    plt.tight_layout()
    plt.show()


def create_immunities_distribution_pie_chart(enemies: pd.DataFrame, **kwargs):
    df_normalized = enemies.explode("immunities").reset_index(drop=True)
    immunities_group = df_normalized.groupby("immunities").size().sort_values()
    plt.figure(figsize=(6, 6))
    plt.pie(immunities_group, labels=immunities_group.index, autopct='%1.1f%%', startangle=90)
    plt.title('Distribution of Immunities')
    plt.tight_layout()
    plt.show()


def create_combined_pie_charts(enemies: pd.DataFrame, **kwargs):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Weaknesses Pie Chart
    df_normalized_weaknesses = enemies.explode("weaknesses").reset_index(drop=True)
    weakness_group = df_normalized_weaknesses.groupby("weaknesses").size().sort_values()
    axes[0].pie(weakness_group, labels=weakness_group.index, autopct='%1.1f%%', startangle=90)
    axes[0].set_title('Distribution of Weaknesses')

    # Resistances Pie Chart
    df_normalized_resistances = enemies.explode("resistances").reset_index(drop=True)
    resistances_group = df_normalized_resistances.groupby("resistances").size().sort_values()
    axes[1].pie(resistances_group, labels=resistances_group.index, autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Distribution of Resistances')

    # Immunities Pie Chart
    df_normalized_immunities = enemies.explode("immunities").reset_index(drop=True)
    immunities_group = df_normalized_immunities.groupby("immunities").size().sort_values()
    axes[2].pie(immunities_group, labels=immunities_group.index, autopct='%1.1f%%', startangle=90)
    axes[2].set_title('Distribution of Immunities')

    # Adjust layout
    plt.tight_layout()
    plt.show()


def create_ability_score_distribution_plot(enemies: pd.DataFrame, **kwargs):
    as_avg = enemies[["str", "dex", "con", "int", "wis", "cha"]].mean()
    enemies['sum_stats'] = enemies[['str', 'dex', 'con', 'int', 'wis', 'cha']].sum(axis=1)
    mean_sum_stats = enemies['sum_stats'].mean().round(2)
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


def create_stats_distribution_plot(enemies: pd.DataFrame, **kwargs):
    # TODO: add movement range, but this can be complicated because enemies can have multiple ways of moving ag. flying, swimming
    enemies["hp_ac_ratio"] = enemies["hp"] / enemies["ac"]
    stats_avg = enemies[["hp", "ac", "hp_ac_ratio"]].mean()
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


def create_character_class_bar_chart(characters: pd.DataFrame, **kwargs):
    character_classes = characters["character_class"].unique()
    sexes = characters["sex"].unique()

    n_classes = len(character_classes)
    n_sexes = len(sexes)
    group_width = 0.8
    bar_width = group_width / n_sexes

    colors = plt.cm.get_cmap('Set1')(np.linspace(0, 1, n_sexes))

    for i, sex in enumerate(sexes):
        sex_data = characters[characters["sex"] == sex]
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


def _create_grouping_pie_chart(df: pd.DataFrame, group_column: str, title: str, legend: bool = True,
                               legend_title: Optional[str] = None, min_percentage=.01, obj=None):
    using_default_plt = False
    if obj is None:
        obj = plt
        using_default_plt = True
    df[group_column] = df[group_column].apply(
        lambda value: value if value != "" else "No " + group_column)
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

    # fig, ax = plt.subplots(figsize=(12, 8))
    wedges, texts, autotexts = obj.pie(values, labels=labels, colors=colors, autopct='%1.1f%%',
                                       pctdistance=0.85, startangle=90)

    # obj.setp(autotexts, size=8, weight="bold")
    # obj.setp(texts, size=10)
    # centre_circle = plt.Circle((0, 0), 0.70, fc='white')
    # obj.gca().add_artist(centre_circle)

    if using_default_plt:
        obj.title(title)
    else:
        obj.set_title(title)
    if legend:
        obj.legend(wedges, labels,
                   title=legend_title,
                   loc="center right"
                   )

    if using_default_plt:
        obj.tight_layout()
        obj.show()


def create_subclasses_bar_chart(characters: pd.DataFrame, obj=None, **kwargs):
    using_default_plt = False
    if obj is None:
        obj = plt
        using_default_plt = True
    all_subclasses = [subclass for subclasses in characters["subclasses"] for subclass in
                      subclasses]
    subclass_counts = pd.Series(all_subclasses).value_counts()

    no_subclass_count = (characters['subclasses'].str.len() == 0).sum()
    no_subclass = "No Subclass"
    subclass_counts[no_subclass] = no_subclass_count

    tick_settings = {"rotation": 45, "ha": "right"}
    if using_default_plt:
        obj.figure(figsize=(12, 8))
        obj.xticks(**tick_settings)

    obj.bar(subclass_counts.index, subclass_counts.values)

    title = 'Distribution of Character Subclasses'
    if using_default_plt:
        obj.title(title)
        obj.tight_layout()
        obj.show()
    else:
        obj.set_title(title)


def create_character_classes_combined_pie_charts(characters: pd.DataFrame, **kwargs):
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    _create_grouping_pie_chart(characters, "character_class", "Character class distribution",
                               legend=False, obj=axes[0])
    _create_grouping_pie_chart(characters, "masterclass", "Masterclass distribution",
                               min_percentage=0, obj=axes[1])
    plt.show()


def create_relationship_web(characters: pd.DataFrame, **kwargs):
    # Initialize data structures
    df_characters = characters
    # df_characters = characters.loc[~characters['name'].isin(['U-Ranos', 'Nayru'])]
    characters = {}
    relationships = []
    output_filename = "data/character_relationships.svg"

    # Iterate over each row in the DataFrame
    for index, row in df_characters.iterrows():
        # Extract character information
        name = str(row['name']).strip()
        surname = str(row['surname']).strip()
        full_name = f"{name} {surname}".strip()
        title = str(row['title']).strip()
        race = str(row['race']).strip()
        sex = str(row['sex']).strip()
        birthday = str(row['birthday']).strip()
        height = str(row['height']).strip()
        functions = row['function']
        character_class = str(row['character_class']).strip()
        subclasses = row['subclasses']
        masterclass = str(row['masterclass']).strip()
        homes = row['homes']
        alignment = str(row['alignment']).strip()
        status = str(row['status']).strip()
        relationships_data = row['relationships']
        lover = str(row['lover']).strip()

        # Store character information
        characters[name] = {
            'name': name,
            'title': title,
            'race': race,
            'sex': sex,
            'alignment': alignment,
            'status': status,
        }

        # Initialize relationships_list
        relationships_list = []

        # Check if relationships_data is not None and not empty
        if relationships_data is not None:
            if isinstance(relationships_data, (list, tuple, np.ndarray, pd.Series)):
                if len(relationships_data) > 0:
                    relationships_list = relationships_data
            elif isinstance(relationships_data, (str, np.str_)):
                relationships_data = relationships_data.strip()
                if relationships_data not in ['', '[]', 'nan']:
                    try:
                        # Replace single quotes with double quotes if necessary
                        relationships_data = relationships_data.replace("'", '"')
                        relationships_list = ast.literal_eval(relationships_data)
                    except (ValueError, SyntaxError):
                        print(f"Error parsing relationships for {name}: {relationships_data}")
            else:
                # relationships_list remains empty
                pass

        # Process relationships_list
        for rel in relationships_list:
            if isinstance(rel, (list, tuple)) and len(rel) == 2:
                rel_name = rel[0].strip()
                rel_desc = rel[1].strip()
                relationships.append((name, rel_name, rel_desc))
            else:
                continue  # Skip invalid entries

        # Parse lover
        if lover not in ['', '[]', 'nan']:
            lover_name = lover.strip()
            relationships.append((name, lover_name, 'Partner'))

    # Build the graph
    G = nx.Graph()

    # Add nodes with attributes
    for char_name, char_attrs in characters.items():
        G.add_node(char_name, **char_attrs)

    # Add edges with relationship types
    for source, target, relation in relationships:
        # Only add edge if both nodes exist
        if source in G.nodes and target in G.nodes:
            G.add_edge(source, target, relation=relation)
        else:
            print(f"One of the nodes '{source}' or '{target}' does not exist in the graph.")

    # Check if graph has nodes and edges
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")

    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        print("The graph is empty. Please check the data parsing logic.")
        return

    # Define edge colors based on relationship types
    edge_colors = {
        'Bruder': 'blue',
        'Schwester': 'pink',
        'Freund': 'green',
        'Freundin': 'green',
        'Partner': 'purple',
        'Vater': 'brown',
        'Mutter': 'brown',
        'Sohn': 'orange',
        'Tochter': 'orange',
        'Cousin': 'cyan',
        'Feind': 'black',
        'Herr': 'red',
        'Ziehvater': 'brown',
        'Ziehmutter': 'brown',
        'Ziehsohn': 'orange',
        # Add other specific relationship types as needed
    }
    default_color = 'gray'

    # Assign colors to edges based on relationship
    edge_colors_list = []
    edge_labels = {}
    for u, v in G.edges():
        relation = G[u][v].get('relation', '')
        # Extract relationship type from relation description
        relation_type = relation.split(',')[0].strip()
        color = edge_colors.get(relation_type, default_color)
        edge_colors_list.append(color)
        edge_labels[(u, v)] = relation  # Store full relation description for edge labels

    # Adjust the layout to spread out nodes
    pos = nx.spring_layout(G, k=2, iterations=200)

    # Increase figure size and resolution
    plt.figure(figsize=(40, 40), dpi=300)

    # Scale node sizes based on degree (number of connections)
    node_sizes = [max(G.degree(n) * 50, 200) for n in G.nodes()]
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue')

    # Draw edges with reduced width
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors_list, width=0.5)

    # Draw edge labels
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='gray', font_size=4)

    # Reduce label font size and adjust label positions
    texts = []
    for node, (x, y) in pos.items():
        texts.append(plt.text(x, y, node, fontsize=6))
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))

    # Get unique relationship types and their colors
    unique_relations = set([rel_type.split(',')[0].strip() for rel_type in edge_colors.keys()])
    legend_handles = []
    for rel_type in unique_relations:
        color = edge_colors.get(rel_type, default_color)
        legend_handles.append(mlines.Line2D([], [], color=color, label=rel_type))

    # Add the legend to the plot
    plt.legend(handles=legend_handles, title='Relationship Types', fontsize=12, title_fontsize=12)

    plt.axis('off')
    plt.title('Character Relationships Clustered by Type', fontsize=15)

    # Save the plot as an SVG file
    output_directory = os.path.dirname(output_filename)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    plt.savefig(output_filename, format='svg', bbox_inches='tight')
    plt.show()
    plt.close()

    print(f"The plot has been saved as {output_filename}")

def create_population_distribution_map(places: pd.DataFrame, markers: pd.DataFrame, **kwargs):
    population_counts = []
    for index, place in places.iterrows():
        marker = markers[markers["name"] == place["name"]]
        if not marker.empty:
            mark = marker.iloc[0]
            population_counts.append((place["name"], place["demography"], mark["lat"], mark["long"]))

    img = plt.imread(os.path.join("data", "faergria-map.png"))
    height, width, _ = img.shape
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(img, extent=(0, width, 0, height))

    for name, count, lat, long in population_counts:
        coeff = 2.9
        x, y = long * coeff, lat * coeff
        circle = plt.Circle((x, y), max(count/500, 5), fill=False, edgecolor='red')
        ax.add_artist(circle)

        ax.annotate(f"{name}: {count}", (x, y), xytext=(5, 5),
                    textcoords="offset points", fontsize=5,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.7))

    ax.set_xlim(0, width)
    ax.set_ylim(0, height)

    # Remove axis ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Add a title
    plt.title("Population Distribution Map")
    plt.savefig(os.path.join("data", "population-distribution-map.png"), dpi=600, bbox_inches="tight")
    plt.show()


def offset_image(y, character_name, ax):
    img_path = os.path.join('data', 'images', f"{character_name.lower()}.png")

    if not os.path.exists(img_path):
        print(f"Image for {character_name} not found at {img_path}")
        return

    img = Image.open(img_path)

    zoom_factor = 0.1

    im = OffsetImage(img, zoom=zoom_factor)
    im.image.axes = ax

    x_offset = -50

    ab = AnnotationBbox(im, (0, y), xybox=(x_offset, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)
    ax.add_artist(ab)


@include_plot
def create_height_distribution_chart(characters: pd.DataFrame, **kwargs):
    characters["height"] = characters["height"].str.replace(',', '.', regex=False)
    characters["height"] = pd.to_numeric(characters["height"], errors="coerce")
    characters = characters.dropna(subset=["height"])
    characters = characters.sort_values(by=["height"])

    fig, ax = plt.subplots(figsize=(10, len(characters) * 0.4))  # Adjust height dynamically

    bar_height = 0.9
    bars = ax.barh(range(len(characters)), characters["height"], height=bar_height, color='skyblue', align='center',
                   alpha=0.8)

    for i, character_name in enumerate(characters["name"]):
        offset_image(i, character_name, ax=ax)

    ax.set_yticks([])  # Hide y-ticks since images are used

    ax.grid(True, which='both', axis='x', linestyle='--', alpha=0.6)
    ax.set_xlabel('Height')
    ax.set_title('Character Height Distribution')

    plt.margins(y=0)  # Eliminate top and bottom margins

    plt.tight_layout()

    plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.01)  # Further reduce top/bottom white space

    plt.show()


'''
WIP Danger Level Calculation
- armor/ac value
- damage per turn capability
- action variety 
- status effects usable and weighting of their strength
- movement range and type with weighting
'''
