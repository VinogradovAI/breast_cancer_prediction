import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import load_data, save_data
from config import TARGET_VARIABLE, TARGET_MASK, CLEANED_DATA_PATH, EDA_PLOTS_PATH, OUTLIERS_LIST

def features_distribution_plot(data, target_column, target_mask, plot_path):
    """
    Plot the distribution of each feature
    :param data: pandas DataFrame
    :param target_column: target column
    :param target_mask: target mask
    """
    fig, axes = plt.subplots(5, 6, figsize=(20, 15))
    axes = axes.ravel()

    for i, column in enumerate(data.columns):
        sns.histplot(data[column], ax=axes[i], kde=True, bins=30, hue=target_column, palette=target_mask)
        axes[i].set_title(column)

    plt.tight_layout()
    plt.show()

    # Write plot to file
    plt.savefig(plot_path + 'features_distribution.png')

def heatmap_plot(data, plot_path):
    """
    Plot the heatmap of the correlation matrix
    :param data: pandas DataFrame
    """
    plt.figure(figsize=(15, 10))
    sns.heatmap(data.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

    # Write plot to file
    plt.savefig(plot_path + 'heatmap.png')

def high_corellation_features(data, threshold):
    """
    Get the features with high correlation
    :param data: pandas DataFrame
    :param threshold: threshold for correlation
    :return: list of features
    """
    corr_matrix = data.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    high_corr_list = [column for column in upper.columns if any(upper[column] > threshold)]

    # VIF calculation
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant

    vif_data = add_constant(data)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(vif_data.values, i) for i in range(vif_data.shape[1])]
    vif["features"] = vif_data.columns

    # Write to file
    with open(EDA_PLOTS_PATH + 'VIF.txt', 'w') as file:
        for i in range(vif.shape[0]):
            file.write(str(vif.iloc[i, 1]) + ' ' + str(vif.iloc[i, 0]) + '\n')

    with open(EDA_PLOTS_PATH + 'high_correlation_features.txt', 'w') as file:
        for feature in high_corr_list:
            file.write(feature + '\n')

    return high_corr_list


def outliers_plot(data):
    """
    Plot the box plots of the features
    :param data: pandas DataFrame
    """
    fig, axes = plt.subplots(5, 6, figsize=(20, 15))
    axes = axes.ravel()

    for i, column in enumerate(data.columns):
        sns.boxplot(data[column], ax=axes[i])
        axes[i].set_title(column)

    plt.tight_layout()
    plt.show()

def outliers_remove(data, OUTLIERS_LIST):
    """
    Remove outliers from the data
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    return data[~((data < (Q1 - 1.5 * IQR)) | (data > (Q3 + 1.5 * IQR)).any(axis=1))]


def outliers_elliptic_envelope(data):
    """
    Remove outliers from the data using Elliptic Envelope
    :param data: pandas DataFrame
    :return: pandas DataFrame
    """
    from sklearn.covariance import EllipticEnvelope

    envelope = EllipticEnvelope(contamination=0.1)
    yhat = envelope.fit_predict(data)
    mask = yhat != -1

    plt.scatter(data.iloc[:, 0], data.iloc[:, 1], c=yhat, cmap='coolwarm')
    plt.show

    plt.savefig(EDA_PLOTS_PATH + 'outliers_elliptic_envelope.png')

    return data[mask]