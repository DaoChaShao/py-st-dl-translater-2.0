#!/usr/bin/env python3.12
# -*- Coding: UTF-8 -*-
# @Time     :   2025/11/15 02:27
# @Author   :   Shawn
# @Version  :   Version 0.1.0
# @File     :   stats.py
# @Desc     :   

from json import load, dump, loads
from numpy import ndarray, array, cumsum, argmax, random as np_random, sum as np_sum, sqrt as np_sqrt
from pandas import DataFrame, Series, read_csv
from pprint import pprint
from pathlib import Path
from random import shuffle
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from torch import Tensor, tensor, float32

from src.utils.decorator import timer

WIDTH: int = 64


class NumpyRandomSeed:
    """ Setting random seed for reproducibility """

    def __init__(self, description: str, seed: int = 27):
        """ Initialise the RandomSeed class
        :param description: the description of a random seed
        :param seed: the seed value to be set
        """
        self._description: str = description
        self._seed: int = seed
        self._previous_np_seed = None

    def __enter__(self):
        """ Set the random seed """
        # Save the previous random seed state
        self._previous_np_seed = np_random.get_state()

        # Set the new random seed
        np_random.seed(self._seed)

        print("*" * 50)
        print(f"{self._description!r} has been set randomness {self._seed}.")
        print("-" * 50)

        return self

    def __exit__(self, *args):
        """ Exit the random seed context manager """
        # Restore the previous random seed state
        if self._previous_np_seed is not None:
            np_random.set_state(self._previous_np_seed)

        print("-" * 50)
        print(f"{self._description!r} has been restored to previous randomness.")
        print("*" * 50)
        print()

    def __repr__(self):
        """ Return a string representation of the random seed """
        return f"{self._description!r} is set to randomness {self._seed}."


@timer
def load_csv(csv_path: str) -> tuple[DataFrame, DataFrame]:
    """ Read data from a dataset file
    :param csv_path: path to the CSV file
    :return: data read from the file
    """
    dataset: DataFrame = read_csv(csv_path)

    y: DataFrame = dataset[:, -1]
    X: DataFrame = dataset.drop(dataset.columns[0], axis=1)

    print(f"X's type is {type(X)}, and its shape is {X.shape}.")
    print(f"y's type is {type(y)}, and its shape is {y.shape}.")

    return X, y


@timer
def load_text(text_data_path: str, cols: bool = False, columns: list | None = None) -> DataFrame:
    """ Read data from a txt file with a structural data format
    :param text_data_path: path to the text data file
    :param cols: whether to specify column names
    :param columns: list of column names
    :return: data read from the text file
    """
    if cols:
        data: DataFrame = read_csv(text_data_path, names=columns, sep=r"\s+")
    else:
        data: DataFrame = read_csv(text_data_path, sep=r"\s+")

    print(f"Loaded text data' shape is {data.shape}.")

    return data


@timer
def summary_dataframe(data: DataFrame) -> None:
    """ Print summary statistics of the data
    :param data: DataFrame containing the data
    """
    print(data.describe())
    print(f"Missing Values: {data.isnull().sum()[data.isnull().sum() > 0]}")
    print(f"Duplicated Rows: {data.duplicated().sum()}")


@timer
def load_paths(base_directory: Path) -> list[tuple[str, int]]:
    """ Load all file paths from a directory
    :param base_directory: path to the directory
    :return: list of file paths in the directory
    """
    data: list[tuple[str, int]] = []

    if base_directory.exists() and base_directory.is_dir():
        for subdir in base_directory.iterdir():
            if subdir.name == "pos" or subdir.name == "neg":
                for file in subdir.iterdir():
                    if file.is_file():
                        label = 1 if subdir.name == "pos" else 0
                        data.append((str(file), label))

        print(f"Loaded {len(data)} files from each directory: {base_directory.name}")
    else:
        print(f"The directory {base_directory} does not exist or is not a directory.")

    return data


@timer
def split_paths(image_paths: list, mask_paths: list, test_size: float = 0.2, shuffle_status: bool = True) -> tuple:
    """ Split the image and mask paths into training and validation sets
    :param image_paths: list of image file paths
    :param mask_paths: list of mask file paths
    :param test_size: the proportion of the dataset to include in the test split
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and validation sets for image and mask paths
    """
    assert len(image_paths) == len(mask_paths), "The number of image paths must be equal to the number of mask paths."

    paired = list(zip(image_paths, mask_paths))
    if shuffle_status:
        shuffle(paired)
        # print(paired)

    split_index: int = int(len(image_paths) * (1 - test_size))
    paired_train_paths: list[tuple[Path, Path]] = paired[:split_index]
    paired_valid_paths: list[tuple[Path, Path]] = paired[split_index:]

    train_image_paths: list[Path] = [image for image, _ in paired_train_paths]
    train_mask_paths: list[Path] = [mask for _, mask in paired_train_paths]
    valid_image_paths: list[Path] = [image for image, _ in paired_valid_paths]
    valid_mask_paths: list[Path] = [mask for _, mask in paired_valid_paths]

    print(f"Training: {len(train_image_paths)} images, {len(train_mask_paths)} masks")
    print(f"Validation: {len(valid_image_paths)} images, {len(valid_mask_paths)} masks")

    return train_image_paths, train_mask_paths, valid_image_paths, valid_mask_paths


@timer
def create_train_valid_split_byXy(
        features: DataFrame | list, labels: DataFrame | list,
        valid_size: float = 0.2, random_state: int = 27, shuffle_status: bool = True
) -> tuple:
    """ Split the data into training and testing sets
    :param features: the DataFrame containing the selected features for training
    :param labels: the DataFrame containing the target masks
    :param valid_size: the proportion of the dataset to include in the test split
    :param random_state: random seed for reproducibility
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and testing sets for features and masks
    """
    X_train, X_valid, y_train, y_valid = train_test_split(
        features, labels,
        test_size=valid_size,
        random_state=random_state,
        shuffle=shuffle_status,
        stratify=None
    )

    if isinstance(X_train, DataFrame):
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_valid shape: {X_valid.shape}, y_valid shape: {y_valid.shape}")
    else:
        print(f"X_train length: {len(X_train)}, y_train length: {len(y_train)}")
        print(f"X_valid length: {len(X_valid)}, y_valid length: {len(y_valid)}")

    return X_train, X_valid, y_train, y_valid


@timer
def create_full_data_split_byXy(
        features: DataFrame | list, labels: DataFrame, randomness: int = 27, shuffle_status: bool = True
) -> tuple:
    """ Split the image and mask paths into training and validation sets
    :param features: DataFrame of features
    :param labels: DataFrame of labels
    :param randomness: random seed for reproducibility
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and validation sets for image and mask paths
    """
    assert len(features) == len(labels), "The number of features must be equal to the number of labels."

    X_train, X_temp, y_train, y_temp = train_test_split(
        features, labels,
        test_size=0.3,
        random_state=randomness,
        shuffle=shuffle_status,
    )
    X_valid, X_prove, y_valid, y_prove = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        random_state=randomness,
        shuffle=shuffle_status,
    )
    print(f"Training set: {X_train.shape}, {y_train.shape}")
    print(f"Validation set: {X_valid.shape}, {y_valid.shape}")
    print(f"Proving set: {X_prove.shape}, {y_prove.shape}")

    return X_train, X_valid, X_prove, y_train, y_valid, y_prove


@timer
def create_full_data_split(data: DataFrame | list, randomness: int = 27, shuffle_status: bool = True) -> tuple:
    """ Split the image and mask paths into training and validation sets
    :param data: DataFrame of features
    :param randomness: random seed for reproducibility
    :param shuffle_status: whether to shuffle the data before splitting
    :return: the training and validation sets for image and mask paths
    """
    data_train, data_temp = train_test_split(
        data,
        test_size=0.3,
        random_state=randomness,
        shuffle=shuffle_status,
    )
    data_valid, data_prove = train_test_split(
        data_temp,
        test_size=0.5,
        random_state=randomness,
        shuffle=shuffle_status,
    )
    print(f"Training set: {len(data_train)}")
    print(f"Validation set: {len(data_valid)}")
    print(f"Proving set: {len(data_prove)}")

    return data_train, data_valid, data_prove


@timer
def save_json(json_data: dict, json_path: str | Path, old_fashioned: bool = False) -> None:
    """ Save a json file
    :param json_data: data to be saved
    :param json_path: path to the json file
    :param old_fashioned: whether to save the old fashioned data
    """
    with open(str(json_path), "w", encoding="utf-8") as file:
        dump(json_data, file, indent=2, ensure_ascii=old_fashioned)

    print(f"JSON data saved to {json_path}.")


@timer
def load_json(json_path: Path) -> dict:
    """ Load JSON data from a file
    :param json_path: path to the JSON file
    :return: data loaded from the JSON file
    """
    data: dict | list | None = None
    if json_path.suffix == ".json":
        with open(str(json_path), "r", encoding="utf-8") as file:
            data: dict = load(file)
    elif json_path.suffix == ".jsonl":
        data: list = []
        with open(str(json_path), "r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    data.append(loads(line))

    print(f"JSON data loaded from {json_path}:")

    return data


@timer
def create_data_transformer(data: DataFrame) -> ColumnTransformer:
    """ Preprocess the data by handling missing values, scaling numerical features, and encoding categorical features.
    :param data: the DataFrame containing the selected features for training
    :return: the fitted ColumnTransformer
    """
    # Divide the columns into numerical and categorical types
    cols_num: list[str] = data.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    cols_cat: list[str] = data.select_dtypes(include=["object", "category"]).columns.tolist()

    # Set a list of transformers to collect the pipelines
    transformers: list[tuple[str, Pipeline, list[str]]] = []

    # Establish a pipe to process numerical features and handle missing values only if they exist
    if cols_num:
        pipe_num = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])
        transformers.append(("num", pipe_num, cols_num))

    # Establish a pipe to process categorical features and handle missing values only if they exist
    if cols_cat:
        pipe_cat = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ])
        transformers.append(("cat", pipe_cat, cols_cat))

    # Establish a column transformer to process numerical and categorical features
    transformer: ColumnTransformer = ColumnTransformer(transformers=transformers)
    # Fit and transform the data
    transformer.fit(data)

    print(f"Preprocessed data type is {type(transformer)}")

    return transformer


@timer
def transform_data(data: DataFrame, preprocessor: ColumnTransformer, is_tensor: bool = False) -> DataFrame | Tensor:
    """ Transform the data using the provided preprocessor"""
    out = preprocessor.transform(data)

    # If the processed data is a sparse matrix, convert it to a dense array
    if hasattr(out, "toarray"):
        out: ndarray = out.toarray()

    # Return DataFrame or Tensor
    if not is_tensor:
        # Rebuild the DataFrame with processed data and proper column names
        output: DataFrame = DataFrame(data=out, columns=preprocessor.get_feature_names_out())
    else:
        # Build the torch tensor with processed data and proper column names
        # - tensor dtype is not quite suitable for PCA
        output: Tensor = tensor(out, dtype=float32)

    print(f"Preprocessed data type is {type(output)}, and its shape: {output.shape}")

    return output


@timer
def pca_importance(data: DataFrame, threshold: float = 0.95, top_n: int = None) -> tuple[list, PCA, DataFrame]:
    """ Calculate PCA feature importance
    :param data: the DataFrame containing the selected features for training
    :param threshold: the cumulative variance ratio threshold to consider
    :param top_n: the number of top important features to return (if None, return all)
    :return: PCA feature importance scores
    """
    # Initialise PCA
    model = PCA()
    model.fit(data)

    # Calculate cumulative variance ratio
    cumulative_variance: ndarray = cumsum(model.explained_variance_ratio_)
    # Determine the number of components to reach the threshold
    n_components: int = int(argmax(cumulative_variance >= threshold) + 1)

    # Build a DataFrame to hold feature loadings
    ratios: DataFrame = DataFrame(
        model.components_.T,
        columns=[f"PC{i + 1}" for i in range(data.shape[1])],
        index=data.columns
    )

    # Calculate the absolute contribution of each feature to the selected components
    ratios["Contribution"] = ratios.iloc[:, :n_components].abs().sum(axis=1)
    # Sort features by their contribution
    ratios: DataFrame = ratios.sort_values("Contribution", ascending=False)
    # print(ratios)

    # Extract important features
    if top_n is not None:
        important_features = ratios.index[:top_n].tolist()
    else:
        important_features = ratios.index[:n_components].tolist()

    print(f"Features meet the threshold of {threshold * 100:.1f}% cumulative variance: {n_components}")
    print("Important Features:")
    pprint(important_features)
    print("Contribution of these features:")
    pprint(ratios.loc[important_features, "Contribution"].values)

    return important_features, model, ratios


@timer
def get_correlation_btw_features(features: DataFrame, top_n: int = 20) -> None:
    """ Display the strongest feature correlations
    :param features: the DataFrame containing the selected features for training
    :param top_n: the number of top correlation pairs to display
    """
    corr_matrix = features.corr()

    # Collect correlation pairs
    corr_pairs: list = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):
            feature_i, feature_ii = corr_matrix.columns[i], corr_matrix.columns[j]
            corr = corr_matrix.iloc[i, j]
            corr_pairs.append((feature_i, feature_ii, corr))

    # Sort correlation pairs by absolute correlation value in descending order
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("Top Feature Correlations:")
    amount: int = len(corr_pairs) if top_n is None else min(top_n, len(corr_pairs))
    for i, (feature_i, feature_ii, corr) in enumerate(corr_pairs[:amount]):
        strength = "★★★" if abs(corr) > 0.7 else " ★★" if abs(corr) > 0.5 else "  ★"
        direction = "↑" if corr > 0 else "↓"
        print(f"{i + 1:03d}. {strength} {feature_i:30s} {direction} {feature_ii:30s} : {corr:6.3f}")


def get_categories_corr_ratio(categories: Series, measurements: Series) -> float:
    """ Correlation Ratio (η) for categorical → numerical association.
    :param categories: categorical variable
    :param measurements: numerical variable
    :return: correlation ratio value
    """
    categories = categories.astype("category")
    cat_values = categories.cat.categories

    means: list[float] = []
    counts: list[int] = []
    for cat in cat_values:
        vals = measurements[categories == cat]
        means.append(vals.mean())
        counts.append(vals.count())

    overall_mean = measurements.mean()
    numerator = np_sum(counts * (array(means) - overall_mean) ** 2)
    denominator = np_sum((measurements - overall_mean) ** 2)

    return np_sqrt(float(numerator) / float(denominator)) if denominator != 0 else 0.0


@timer
def get_correlation_btw_Xy(X: DataFrame, y: Series, top_n: int = 20, threshold: float = 0.05) -> list[str]:
    """ Get the correlation of features with the label
    :param X: the DataFrame containing the selected features for training
    :param y: the Series containing the target values
    :param top_n: the number of top correlation pairs to display
    :param threshold: the minimum absolute correlation value to consider
    """
    # Divide the columns into numerical and categorical types
    cols_num: list[str] = X.select_dtypes(include=["int32", "int64", "float32", "float64"]).columns.tolist()
    cols_cat: list[str] = X.select_dtypes(include=["object", "category"]).columns.tolist()

    correlations: list[tuple[str, float]] = []

    # Calculate correlations of numerical features with the label
    for col in cols_num:
        corr = X[col].corr(y)
        correlations.append((col, corr))
    # Calculate correlations of categorical features with the label
    for col in cols_cat:
        corr = get_categories_corr_ratio(X[col], y)
        correlations.append((col, corr))

    # Sort correlations by absolute correlation value in descending order
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)

    print("Top Feature Correlations with Label:")
    amount = len(correlations) if top_n is None else min(top_n, len(correlations))
    for i, (feature, corr) in enumerate(correlations[:amount], 1):
        strength = "★★★" if abs(corr) > 0.7 else " ★★" if abs(corr) > 0.5 else "  ★"
        direction = "↑" if corr > 0 else "↓"
        print(f"{i:03d}. {strength} {feature:30s} {direction} : {corr:6.3f}")

    return [name for name, corr in correlations if abs(corr) >= threshold][:top_n]


if __name__ == "__main__":
    pass
