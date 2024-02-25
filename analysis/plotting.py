import pandas as pd
import matplotlib.pyplot as plt


def get_scores(df, property):
    return df[property]


def get_overlayed_distributions(
    sources: list[str],
    shared_pre_dir="../output/",
    bar_colours=["blue", "red"],
    property="score",
):
    """
    By default handles two different sources. Plot an overlayed graph of both.

    sources : A list of filenames
    shared_pre_dir : The relative path that all the sources share
    bar_colours : A list of strings of the colour of each graph
    """
    for source, colour in zip(sources, bar_colours):
        output = pd.read_csv(shared_pre_dir + source)
        plt.hist(get_scores(output, property), color=colour, alpha=0.4)

    plt.legend(["1", "2"])

    return plt


def get_scatter_distributions_comparison(
    sources: list[str],
    shared_pre_dir="../output/",
    property="score",
):
    """
    By default handles two different sources. Plot an overlayed graph of both.

    sources : A list of filenames
    shared_pre_dir : The relative path that all the sources share
    """
    scatter_pair = []
    for source in sources:
        output = pd.read_csv(shared_pre_dir + source)
        scatter_pair.append(get_scores(output, property))

    plt.scatter(*scatter_pair, alpha=0.4)

    return plt


def get_unique_output(input_id, folder="../output", property="score"):
    """
    Assuming you only have one input with a certain id, get the output, from the output folder
    """
    extra = ""
    found_filenames = []

    if property == "auc":
        extra = "auc"

    import os

    for file in os.listdir(folder):
        if file.startswith("output_" + input_id + extra + "-"):
            found_filenames.append(os.path.join(file))

    if len(found_filenames) > 1:
        print("Multiple matches were found, so this function won't work as intended")
        raise ValueError("Multiple matches found, when we assumed uniqueness")

    if len(found_filenames) == 0:
        raise ValueError("No file found.")

    return found_filenames[0]
