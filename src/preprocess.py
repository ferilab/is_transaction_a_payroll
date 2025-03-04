import pandas as pd
import os
import sys
    ### For working with strings, timing, file saving/loading
import re
import joblib

    # To encode categorical variables
from sklearn.preprocessing import LabelEncoder

    # First set the absolute path to the project root (for proper imports)
sys.path.append(os.path.abspath(".."))
    # All utility functions are in utils
import src.utils as utils


def preprocess_data() -> str:
        # Also, get the project root path (one level up from dev_and_run folder) to navigate through other folders like data
    project_root = os.path.abspath("..")

    file_path = os.path.join(project_root, "data/")
    data_df = pd.read_csv(file_path + 'train.csv')

        # 1. Data preparartion
        # 1.1 First, let's remove the wrongly labled records
    data_df, wrongly_labelled = utils.remove_wrongly_labelled_data(data_df)
    print(f"{len(wrongly_labelled)} records where removed for being wrongly labelled.")

        # 1.2 Convert date to more meaningful variables
    data_df = utils.add_date_info(data_df)

        # 1.3 Trim descriptions from unnecessary and useless transaction numbers
        # List to store extracted transaction numbers
    trans_numbers = []

        # Function to remove trailing transaction numbers and store them in a variable
    def prune_trans_number(description, trans_numbers):
        
            # Find transaction numbers with the format "123-43-2850" at the end of the string
        match = re.search(r'\b\d{3}-\d{2}-\d{4}$', description)
        if match:
            trans_numbers.append(match.group())  # Store the found transaction number
                # Remove the transaction number from the description
            return re.sub(r'\b\d{3}-\d{2}-\d{4}$', '', description).strip()
        
        return description
        
        # We'll add a new column for the trimmed description because we need the original one for further analysis later 
    data_df['description_trimmed'] = data_df['description'].apply(prune_trans_number)

        # Here we check to make sure that our guess about the unique nature of transaction numbers is true.  
    unique_trans_numbers = set(trans_numbers)
    print(f"Total and unique transaction numbers found: {len(trans_numbers), len(unique_trans_numbers)}")

        # Prepare the data (There are 5726 unique values in description_trimmed)

        # 1.4 Encode the categorical variables (actually)
    label_encoder = LabelEncoder()
    data_df['description_trimmed_encoded'] = label_encoder.fit_transform(data_df['description_trimmed'])

    file_name = 'train_preprocessed.csv'
    data_df.to_csv(file_path + file_name)
    
    return file_name