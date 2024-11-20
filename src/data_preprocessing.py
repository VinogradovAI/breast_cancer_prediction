import pandas as pd
import io

from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, LOG_RAW_DATA_INFO_PATH, TARGET_VARIABLE, REMOVAL_COLUMNS
from src.utils import load_data, save_data

def info_data(data):
    """
    Display information about the data
    :param data: pandas DataFrame
    """
    print(data.info())
    print(data.describe())
    print(data.head().round(2).T)
    print(data.columns)


def log_data_info(data, log_file_path):
    """
    Log information about the data to a file
    :param data: pandas DataFrame
    :param log_file_path: path to the log file
    """
    with open(log_file_path, 'w') as log_file:
        buffer = io.StringIO()
        data.info(buf=buffer)
        data_info = buffer.getvalue()

        data_description = data.describe().round(2).T.to_string()
        data_head = data.head().round(2).T.to_string()
        data_duplicates = data.duplicated().sum()
        data_nulls = data.isnull().sum().sum()
        data_unique = data.nunique()
        data_target_distribution = data[TARGET_VARIABLE].value_counts()

        log_file.write("Raw Data Info:\n")
        log_file.write(data_info + "\n\n")
        log_file.write("\nRaw Data Description:\n")
        log_file.write(data_description + "\n\n")
        log_file.write("\nRaw Data Head:\n")
        log_file.write(data_head + "\n\n")
        log_file.write("\nRaw Data Duplicates:\n")
        log_file.write(str(data_duplicates) + "\n\n")
        log_file.write("\nRaw Data Nulls:\n")
        log_file.write(str(data_nulls) + "\n\n")
        log_file.write("\nRaw Data Unique Values:\n")
        log_file.write(str(data_unique) + "\n\n")
        log_file.write("\nRaw Data Target Variance Distribution:\n")
        log_file.write(str(data_target_distribution) + "\n\n")


def remove_columns(data, columns):
    """
    Remove columns from the data
    :param data: pandas DataFrame
    :param columns: list of columns to remove
    :return: pandas DataFrame
    """
    return data.drop(columns=columns)


def zero_variance(data):
    """
    Check and if % of total raws with zero values is less than 5% from total number of raws then remove raws with zeros
    else give a warning
    """
    zero_values = (data == 0).sum(axis=1)
    zero_values_percent = zero_values / data.shape[1] * 100
    zero_values_percent = zero_values_percent[zero_values_percent > 0]
    if zero_values_percent.shape[0] / data.shape[0] < 0.05:
        data = data.drop(zero_values_percent.index)
    else:
        print("Warning: More than 5% of the data contains zeros")

    return data


# Main preprocessing function
def main():
    # Load raw data
    raw_data = load_data(RAW_DATA_PATH)

    # Log raw data info
    log_data_info(raw_data, LOG_RAW_DATA_INFO_PATH)

    # Remove unnecessary columns
    cleaned_data = remove_columns(raw_data, REMOVAL_COLUMNS)

    # Zero variance
    cleaned_data = zero_variance(cleaned_data)

    # Save cleaned data
    save_data(cleaned_data, CLEANED_DATA_PATH)

# Preprocessing Execution
if __name__ == "__main__":
    main()
