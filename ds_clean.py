import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import re  # regular expression

def clean_values(df, 
                 mapping = {
                    "ok": 1,
                    "oui": 1,
                    "non": 0,
                    "no": 0,
                    }
                 ):
    #https://note.nkmk.me/en/python-pandas-map-replace/
    
    # Remove trailing spaces from each cell
    # df = df.map(lambda x: x.rstrip() if isinstance(x, str) else x)
    #df = df.replace(r"^\s+|\s+$", "", regex=True)
    
    # Function to replace values in a case-insensitive manner
    def case_insensitive_replace(value):

        if isinstance(value, str):
            return mapping.get(value.lower(), value)
        return value

    # Apply the function to each element in the DataFrame
    df = df.map(case_insensitive_replace)

def regex_column_selector(df, regex_list):
    """like make_column_selector, but using regex list
    
    equivalent to: make_column_selector(pattern=r"(?i)date")(df)

    Args:
        df (_type_): _description_
        regex_list (_type_): [ r"var1", r"(?i)VaR2", # (?i) case-insensitive]

    Returns:
        _type_: list of selected columns
    """
    columns_list = []

    for column in df.columns:
        if any(re.search(indicator, column) for indicator in regex_list):
            columns_list.append(column)

    return columns_list


#date_columns = regex_column_selector(df, [r"(?i)date"])
#print("Identified date columns:", date_columns)

# clean boolean and category

# Identify columns to convert to boolean by checking column names for specific patterns
def clean_boolean_columns(df, regex_bool = [r"0/1", r"=1", r"1=", r"≥ 1"], columns_to_append = ["BAI"]):
    # Identify columns to convert to boolean by checking column names for specific patterns
    
    bool_columns = regex_column_selector(df, regex_bool)

    # Append additional columns directly
    bool_columns.extend(columns_to_append)
    print(bool_columns)
    
    #to manage non boolean values
    def convert_to_boolean(value):
        if pd.isna(value):
            return np.nan
        elif value == 0 or value == "0":
            return False
        elif value == 1 or value == "1":
            return True
        else:
            return np.nan


    # Apply the custom function to the identified columns
    df[bool_columns] = df[bool_columns].map(convert_to_boolean)

    # Convert to boolean dtype
    df[bool_columns] = df[bool_columns].astype("boolean")
    
    unique_bool = df[df.select_dtypes(include=["boolean"]).columns].nunique()

    print(unique_bool[unique_bool <= 1])

    # Remove columns with only one unique category
    df = df.drop(columns=unique_bool[unique_bool <= 1].index)
    
    return df
# clean numerical

def clean_numerical_columns(df,
                            # Identify columns to convert to numeric by checking column names for specific patterns
                            regex_float = [r"\(L\)",
                                           r"(?i)g\s*/\s*[d]?L",]  # Matches g/L or g/dL, (?i) case-insensitive with optional spaces
                            ):
    
    unique_object = df[df.select_dtypes(include=["object"]).columns].nunique()

    display(
        pd.DataFrame(unique_object[unique_object > 20])
    )  # columns may be numerical, with too many unique values to be categorical
    
    
    

    columns_float = regex_column_selector(df, regex_float)
    print("Identified float columns:", columns_float)
    
    # in float colomns, clean numbers with double like ,, or ..
    df[columns_float] = df[columns_float].replace(
        r"^(\d+)[.,]{1,2}(\d+)", r"\1.\2", regex=True
    )

    # regex: take the first number, then any non digit character surrounded or not by comma, then the second number
    df[columns_float] = df[columns_float].replace(
        r"^(\d+),{0,1}[^0-9.,],{0,1}(\d+)", r"\1.\2", regex=True
    )

    # Function to convert '1/80' like strings to numeric
    def convert_to_numeric(x):
        """Convert a string to a float if it contains a slash character.
        WARNING: aggresive conversion, may lead to NaN values if the string is not a fraction.

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        if isinstance(x, str) and "/" in x:
            num, denom = x.split("/")
            return float(num) / float(denom)
        return None  # Handle NaN or None


    # Apply the conversion function to the column
    #df["AAN Titre"] = df["AAN Titre"].map(convert_to_numeric)
    
    # force conversion of float columns to numeric
    for col in columns_float:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def extract_non_float_rows(df):
    """Extract rows where at least one value in the row is not a float.
    Usage: print(len(extract_non_float_rows(df[columns_float])))

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    # Fonction pour vérifier si une valeur peut être convertie en float
    def is_float(x):
        try:
            float(x)
            return True
        except ValueError:
            return False

    # Créer un masque booléen pour les lignes à conserver
    mask = np.zeros(len(df), dtype=bool)

    for col in df.columns:
        # Ajouter au masque les lignes où la valeur n'est pas un float
        mask |= ~df[col].astype(str).apply(is_float)

    # Extraire les lignes qui correspondent au masque
    result = df[mask]

    return result