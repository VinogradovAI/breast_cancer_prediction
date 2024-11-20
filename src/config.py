PROJECT_PATH = 'D:/ml_school/Skillfactory/team_cases/breast_cancer_prediction'

# Paths
RAW_DATA_PATH = PROJECT_PATH + "/data/raw/data.csv"
CLEANED_DATA_PATH = PROJECT_PATH + '/data/processed/cleaned.csv'
FEATURE_PATH = PROJECT_PATH + '/data/processed/features.csv'
LOG_RAW_DATA_INFO_PATH = PROJECT_PATH + '/logs/preprocessing_log.txt'
EDA_PLOTS_PATH = PROJECT_PATH + '/logs/EDA_plots/'

TARGET_VARIABLE = 'Diagnosis'
TARGET_MASK = {'M': 1, 'B': 0}

REMOVAL_COLUMNS = ['ID']
OUTLIERS_LIST = []



