from lib.imports import *

class Preprocessing:
    """
    This class provides methods for data preprocessing related to waste management and route optimization.
    """
    
    def add_process_frequency_data(self, data: pd.DataFrame = None, 
                                   path="data/input_data/czetotliwosci_one-hot-encoder.xlsx"):
        """
        Reads and processes frequency data from an Excel file and optionally merges it with an existing DataFrame.
    
        Parameters:
        - data (pd.DataFrame or None): Existing DataFrame. If provided, the frequency data is merged with it.
        - path (str): Path to the Excel file. Default is "data/czetotliwosci_one-hot-encoder.xlsx".
    
        Returns:
        - merged_data (pd.DataFrame): Merged and processed DataFrame.
        """
    
        # Read frequency data from Excel
        process_frequency_data = pd.read_excel(path).fillna(0)
   
        # Convert 'Numer tygodnia' column to string
        process_frequency_data['Numer tygodnia'] = process_frequency_data['Numer tygodnia'].astype(str)
  
        # Apply a function to create combinations of 'Numer tygodnia' and 'Dni tygodnia' columns
        process_frequency_data['Częstotliwość'] = process_frequency_data.apply(
            lambda row: ', '.join(''.join(map(str, p)) if p[0] != "niestandardowe" else "niestandardowe"
                                  for p in product(row['Numer tygodnia'].split(', '), row['Dni tygodnia'].split(', '))),
            axis=1
        )
  
        if isinstance(data, pd.DataFrame):
            # Merge data based on 'Cykl odbiorów' and 'Cykl' columns
            df = data.copy()
            merged_data = df.merge(process_frequency_data, left_on="Cykl odbiorów", right_on="Cykl", how="left")
        else:
            return process_frequency_data
        
        # Split the 'Częstotliwość' column into lists
        merged_data['Częstotliwość'] = merged_data['Częstotliwość'].apply(lambda x: x.split(",") if pd.notnull(x) else [])

        # Explode the 'Częstotliwość' column into separate rows
        merged_data = merged_data.explode('Częstotliwość').reset_index(drop=True)
 
        return merged_data

    def merge_fractions(self, data: pd.DataFrame, old_label: list[list[int]] = None):
        """
        Merges fractions in the DataFrame based on specified old labels.
    
        Parameters:
        - data (pd.DataFrame): Input DataFrame.
        - old_label (list of list of int or None): List of old labels to be merged. If None, it uses linkable combinations.
    
        Returns:
        - data (pd.DataFrame): Modified DataFrame with merged fractions.
        """
    
        # Create a copy of the input DataFrame
        data = data.copy()
    
        # Convert 'Kod odpadu' to string if not already and extract the first part before the dot "."
        if type(data['Kod odpadu'][0]) != str:
            data['Kod odpadu'] = data['Kod odpadu'].astype(str).apply(lambda x: x.split('.')[0] if isinstance(x, str) else x)
    
        # If old labels are not provided, use linkable combinations
        if not old_label:
            old_label = self.support_linkable_combinations()
    
        # Replace old labels with the new label in the 'Kod odpadu' column
        for label_list in old_label:
            new_label = ','.join([str(elem) for elem in label_list]).replace(",", "_")
            for label in label_list:
                data.loc[data['Kod odpadu'].astype(str) == str(label), "Kod odpadu"] = new_label
                
        # Replace commas with underscores in the 'Kod odpadu' column
        # data['Kod odpadu'] = data['Kod odpadu'].str.replace(", ", "_")
        
        return data
    
    def divide_into_groups(self, data: pd.DataFrame, return_group_name=False, save=False, folder_path="data\sorted_by_groups"):
        """
        Divides a DataFrame into groups based on the 'Rodzaj klienta', 'Kod odpadu', and 'day' columns.
    
        Args:
            dataframe (pd.DataFrame): Input data to be divided into groups.
            return_group_name (bool, optional): Flag indicating whether to return group names. Default is False.
            save (bool, optional): Flag indicating whether to save data to CSV files. Default is False.
            folder_path (str, optional): Path to the folder where the data should be saved. Default is "data\sorted_by_groups".
    
        Returns:
            Union[pd.DataFrame, Tuple[pd.DataFrame, list]]: DataFrame with the division into groups or DataFrame and a list of group names if return_group_name is True.
    
        This function creates a 'groups' column by combining 'Rodzaj klienta', 'Kod odpadu', and 'day'. It then divides the data into groups based on the created column. It can save the divided data to CSV files.
        """
    
        # Create a copy of the input data
        df = data.copy()
        
        # Create a dictionary mapping 'Rodzaj klienta' to unique groups
        try:
            unique_customer_type = {}
            for i in pd.unique(df["Rodzaj klienta"]):
                if isinstance(i, str) and i[:1] not in ["B", "P"]:
                     unique_customer_type[i] = str(i)
                else:
                    if pd.isna(i):
                        unique_customer_type[i] = i 
                    else:
                        unique_customer_type[i] = str(i)[:1]
        
        
            df["Częstotliwość"] = df["Częstotliwość"].astype(str).str.strip()
            df['Kod odpadu'] = df['Kod odpadu'].astype(str).str.strip()
            # Create a 'groups' column by combining 'Rodzaj klienta', 'Kod odpadu', and 'day'
            df["groups"] = df["Częstotliwość"] + "_" +\
                           df["Rodzaj klienta"].map(unique_customer_type).astype(str) + "_" + \
                           df['Kod odpadu'].str.replace("_ ", "_")
            
            df["Częstotliwość"] = df["Częstotliwość"].str.strip()
                           
        except KeyError as e:
            print( f"Missing column {e} for calculating data" )
            return None
    
        # Get unique group names
        group_names = list(pd.unique(df["groups"]))
    
        # Save data to CSV files
        if save:
            folder_path = folder_path
            os.makedirs(folder_path, exist_ok=True)
            for i in group_names:
                file_path = os.path.join(folder_path, f"{i}.csv")
                save_data = df[df["groups"] == i]
                save_data.to_csv(file_path, index=False)
                
            file_path = os.path.join(folder_path, "group_name.pkl")    
            with open(file_path, 'wb') as file:
                pickle.dump(group_names, file)
        
        # Return data or data and a list of group names
        if return_group_name:
            return df, group_names
        else:
            return df
    
    
        
        
        # Get unique group names
        group_names = list(pd.unique(df["groups"]))
    
        # Save data to CSV files
        if save:
            folder_path = folder_path
            os.makedirs(folder_path, exist_ok=True)
            for i in group_names:
                file_path = os.path.join(folder_path, f"{i}.csv")
                save_data = df[df["groups"] == i]
                save_data.to_csv(file_path, index=False)
                
            file_path = os.path.join(folder_path, "group_name.pkl")    
            with open(file_path, 'wb') as file:
                pickle.dump(group_names, file)
        
        # Return data or data and a list of group names
        if return_group_name:
            return group_names
        else:
            return df
    
    def support_linkable_combinations(self, df=None, patch="data/input_data/odpady_kompresja.csv"):
        """
        Extracts linkable combinations from a DataFrame or CSV file.
    
        Parameters:
        - df (pd.DataFrame or None): Input DataFrame. If None, it reads the data from the specified CSV file.
        - patch (str): Path to the CSV file. Default is "data/odpady_kompresja.csv".
    
        Returns:
        - linkable_combinations (list): List of linkable combinations.
        """
    
        # Read DataFrame from CSV file if not provided
        if not df:
            types = {"Kod odpadu": str, "Możliwość łączenia": str}
            df = pd.read_csv(patch, dtype = types)
    
    
        df["Kod odpadu"] = df["Kod odpadu"].str[:-2]
        df.loc[:, "temporary"] = df["Kod odpadu"] +","+ df["Możliwość łączenia"]
        df = df.dropna()
        df["temporary"] = df["temporary"].str.replace(r' ', '').str.split(",")
        
        return df["temporary"].to_list()

    def process_car_data(self, data: pd.DataFrame = None, path = "data/input_data/samochody.csv"):
        """
        Processes a DataFrame related to cars and their specifications.
    
        Parameters:
        - data (pd.DataFrame or None): Input DataFrame. If provided, the function processes the given DataFrame; otherwise, it reads data from the specified CSV file.
        - path (str): Path to the CSV file. Default is "data/samochody_v2.csv".
    
        Returns:
        - df (pd.DataFrame): Processed DataFrame.
        """
        if isinstance(data, pd.DataFrame):
            # Create a copy of the input DataFrame
            df = data.fillna(0).copy()
        else:
            # Read data from the specified CSV file and fill NaN values with 0
            df = pd.read_csv(path).fillna(0)
    
        # Set index based on specified columns and drop a specific row
        df = df.set_index(['Samochód', 'pojemność', 'KOM_1', 'KOM_2', 'Ilość komór', "Zagniatarka"])
        df = df.drop("L3500", axis=1)
    
        # Get column names
        columns = df.columns
    
        # Initialize a new column "Typ pojemnika"
        df.loc[:, "Typ pojemnika"] = "0"
        
        # Iterate through rows and populate "Typ pojemnika" based on non-zero values
        for index, series in df.iterrows():
            value = ""
            for i, v in enumerate(series):
                if v != "0" and v != "0.0":
                    value += "," + columns[i]
            df.at[index, "Typ pojemnika"] = value
    
        # Split the values in "Typ pojemnika" column into a list
        df.loc[:,'Typ pojemnika'] = df['Typ pojemnika'].apply(lambda x: x.split(",") )
    
        # Reset index and explode "Typ pojemnika" column into separate rows
        df = df.reset_index()
        df = df.explode('Typ pojemnika').reset_index(drop=True)
    
        # Remove rows where "Typ pojemnika" is an empty string
        empty_string_rows = df.apply(lambda x: x == '').any(axis=1)
        df = df[~empty_string_rows]
    
        # Replace "MPC" with "MPC-5" in "Typ pojemnika" column
        df.loc[df["Typ pojemnika"] == "MPC", "Typ pojemnika"] = "MPC-5"
    
        # Drop duplicate rows
        df = df.drop_duplicates()
    
        return df

    def added_container_types(self, data: pd.DataFrame):
        """
        Adds container type information to the input DataFrame.

        Parameters:
        - data (pd.DataFrame): Input DataFrame containing waste routing data.

        Returns:
        pd.DataFrame: Updated DataFrame with container type information added.
        """
        typy_pojemnika = pd.read_csv(r"data/input_data/typy_pojemnik.csv", sep=";")
        data = data.merge(typy_pojemnika[["Typ_pojemnika", "Pojemnosc", "Typ_wlasciwy"]], 
                          left_on="Typ pojemnika", right_on="Typ_pojemnika", how="left")
        return data

    def added_waste_compression(self, data: pd.DataFrame):
        """
        Adds waste compression information to the input DataFrame.

        Parameters:
        - data (pd.DataFrame): Input DataFrame containing waste routing data.

        Returns:
        pd.DataFrame: Updated DataFrame with waste compression information added.
        """
        waste_compression = pd.read_csv("data/input_data/odpady_kompresja.csv") 
        data = data.merge(waste_compression, left_on="Kod odpadu", right_on='Kod odpadu', how="left")
        return data 

    def main_for_route(self, data: pd.DataFrame):
        """
        Perform the main data processing for route optimization.
    
        Parameters:
        - data (pd.DataFrame): Input DataFrame containing relevant data.
    
        Returns:
        - processed_data (pd.DataFrame): Processed DataFrame for route optimization.
        """
    
        # Step 1: Merge additional information about container types and waste compression
        data = self.added_waste_compression(data)
        data = self.added_container_types(data)
        
        # Step 2: Adjust container capacity based on waste compression factor
        data.loc[:, "Pojemnosc"] = data["Pojemnosc"] * data["Kompresja"]
        
        # Step 3: Add and process frequency data
        data = self.add_process_frequency_data(data)
        
        # Step 4: Merge fractions based on specified labels or linkable combinations
        data = self.merge_fractions(data)
        
        # Step 5: Divide data into groups based on specific columns
        data = self.divide_into_groups(data, return_group_name=False, save=True)
        
        # Step 6: Drop unnecessary columns
        drop_columns = ["Kompresja", "Nazwa_odpadu", "Typ pojemnika", "Typ_wlasciwy", "Typ_pojemnika", 
                        "Cykl odbiorów", "Możliwość łączenia"]
        data = data.drop(drop_columns, axis=1)
    
        return data



# drot funktions
def address_collector(data: pd.DataFrame, columns_for_address: list = ["Kod pocz.", "Miasto", "Ulica", "Nr budynku"]) -> pd.DataFrame:
    """
    Creates a new column 'full address' by concatenating specified columns to form an address.

    Parameters:
    - data (pd.DataFrame): Input DataFrame containing address-related columns.
    - columns_for_address (list): List of column names used to construct the address. Defaults to ["Kod pocz.", "Miasto", "Ulica", "Nr budynku"].

    Returns:
    pd.DataFrame: DataFrame with the new 'full address' column.
    """
    try:
        # Create a copy of columns with data to create an address
        df = data[columns_for_address].copy()
    except KeyError as e:
        print(f"Missing column {e} to create the address")
        return None

    # Ensure that missing values are replaced with empty strings
    df = df.fillna('')

    # Insert the "Country" column with a constant value "Poland"
    df.insert(0, 'Country', 'Poland')

    # Create a new column "fuul_address" containing concatenated addresses
    data.loc[:, 'full address'] = df.apply(lambda row: ', '.join(row.astype(str)), axis=1)

    # Remove trailing commas and spaces at the beginning and end of each line
    data.loc[:,'full address'] = data['full address'].str.strip(', ')

    # Return DataFrame with the new "fuul_address" column
    return data











