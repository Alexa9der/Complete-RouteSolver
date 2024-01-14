from lib.imports import *
from project_functions.visualization import colored_marker_map
# Grocoder
class Geocoder:
    """
    The Geocoder class provides a simple interface for geocoding addresses using the Nominatim OpenStreetMap API.

    Parameters:
    - `max_threads` (int): Maximum number of concurrent threads.

    Attributes:
    - `base_url` (str): Base URL for the Nominatim OpenStreetMap API.
    - `headers` (dict): HTTP headers for the API request.
    - `max_threads` (int): Maximum number of concurrent threads for geocoding.
    - `geocoded_results` (dict): Cache for storing geocoding results.
    - `geo_data_file` (str): Path to the CSV file for storing geocoding results.

    Methods:
    - `geocode_address(address: str) -> dict`: Geocodes a single address.
    - `geocode_addresses(df: DataFrame)`: Geocodes a list of addresses from a DataFrame.
    - `get_results() -> list`: Retrieves a list of geocoding results.
    - `save_results_to_csv(directory: str = None, filename: str = None)`: Saves geocoding results to a CSV file.
    - `address_collector(data: DataFrame, columns_for_address: list = ["Kod pocz.", "Miasto", "Ulica", "Nr budynku"]) -> DataFrame`:
        Generates a new column "full_address" in the DataFrame by concatenating specified address elements.

    Private Methods:
    - `_send_geocode_request(address: str) -> dict`: Sends a request to the geocoding server.

    Example Usage:
    ```python
    # Create a Geocoder instance
    geocoder = Geocoder(max_threads=10)

    # Geocode a single address
    result = geocoder.geocode_address("123 Main St, City, Country")

    # Geocode a DataFrame of addresses
    df = pd.DataFrame({"Kod pocz.": [12345, 54321], "Miasto": ["City1", "City2"], "Ulica": ["Street1", "Street2"], "Nr budynku": [1, 2]})
    geocoder.geocode_addresses(df)

    # Retrieve and print geocoding results
    results = geocoder.get_results()
    print(results)

    # Save geocoding results to a CSV file
    geocoder.save_results_to_csv(directory="data\\geocoding_results", filename="geocoded_data.csv")
    ```

    Note: Make sure to replace "YourUserAgent" in the `headers` attribute with a valid user agent string.
    """

    def __init__(self, max_threads=10):
        """
        Initializes a Geocoder object.

        Args:
        max_threads (int): Maximum number of concurrent threads.
        """
        self.base_url = "https://nominatim.openstreetmap.org/search"
        self.headers = {"User-Agent": "YourUserAgent"} 
        self.max_threads = max_threads
        self.geocoded_results = {}
        self.geo_data_file = r"data\\data_after_geocoding\\geodata.csv"

    def geocode_address(self, address):
        """
        Geocodes the given address.

        Args:
        address (str): Address to geocode.

        Returns:
        dict: Geocoding result (dictionary with information about the geocoded address).
        """
        # Check if a file with geographic data exists and if the address is already in the file
        if os.path.exists(self.geo_data_file):
            with open(self.geo_data_file, "r", encoding="utf-8") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    if row["address"] == address:
                        # Address already exists in the file, return the result
                        result = {
                            "address": address,
                            "longitude": row["longitude"],
                            "latitude": row["latitude"]
                        }
                        self.geocoded_results[address] = result  # Add the result to geocoded_results
                        return result

        # Address not found in the file or the file does not exist, send a request to the server
        result = self._send_geocode_request(address)
        
        # Add the result to geocoded_results
        self.geocoded_results[address] = result

        return result

    def geocode_addresses(self, df):
        """
        Geocodes a list of addresses.

        Args:
        df (DataFrame): DataFrame containing addresses.
        """
        with concurrent.futures.ThreadPoolExecutor(self.max_threads) as executor:
            future_to_address = {
                executor.submit(self.geocode_address, row["full address"]): row["full address"]
                for index, row in df.iterrows()
            }

            for future in concurrent.futures.as_completed(future_to_address):
                address = future_to_address[future]
                result = future.result()

        print("Geocoding completed.")

    def get_results(self):
        """
        Retrieves a list of geocoding results.

        Returns:
        list: List of dictionaries with information about geocoded addresses.
        """
        return list(self.geocoded_results.values())

    def save_results_to_csv(self, directory=None, filename=None):
        """
        Saves geocoding results to a CSV file.

        Args:
        directory (str): Directory where the file should be saved.
        filename (str): Name of the file where the results should be saved.
        """
        if not directory:
            directory = "data\\data_after_geocoding"
        if not filename:
            filename = "geodata.csv"
        # Create the directory if it does not exist
        os.makedirs(directory, exist_ok=True)

        # Full path to the CSV file
        csv_path = os.path.join(directory, filename)

        # Check if data already exists in the file
        existing_data = {}
        if os.path.exists(csv_path):
            with open(csv_path, "r", newline="", encoding="utf-8") as csv_file:
                csv_reader = csv.DictReader(csv_file)
                for row in csv_reader:
                    existing_data[row["address"]] = row

        # If data already exists in the file, do not save it again
        written_addresses = set()
        for result in self.geocoded_results.values():
            address = result["address"]
            if address not in written_addresses and address not in existing_data:
                with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
                    fieldnames = ["address", "longitude", "latitude"]
                    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

                    # Add headers if the file is empty
                    if os.path.getsize(csv_path) == 0:
                        writer.writeheader()

                    writer.writerow(result)
                written_addresses.add(address)

        print(f"Results saved to {csv_path}.")

    def _send_geocode_request(self, address):
        """
        Sends a request to the geocoding server.

        Args:
        address (str): Address to geocode.

        Returns:
        dict: Geocoding result (dictionary with information about the geocoded address).
        """
        params = {"format": "json", "q": address}
        response = requests.get(self.base_url, params=params, headers=self.headers)
        data = response.json()

        if data:
            latitude = float(data[0]["lat"])
            longitude = float(data[0]["lon"])
            result = {"address": address, "longitude": longitude, "latitude": latitude}
            self.geocoded_results[address] = result  # Save the result in the cache
            return result
        else:
            result = {"address": address, "longitude": None, "latitude": None}
            self.geocoded_results[address] = result  # Save the result in the cache
            return result

    def address_collector(self, data, columns_for_address: list = ["Kod pocz.", "Miasto", "Ulica", "Nr budynku"]) -> pd.DataFrame:
        """
        This function takes a DataFrame containing address elements and generates a new column "fuul_address"
        containing concatenated addresses based on the specified columns.
    
        Parameters:
        data (pd.DataFrame): DataFrame containing columns with address elements.
        columns_for_address (list, optional): List of column names containing address elements. Default is ["Kod_pocz.", "Miasto", "Ulica", "Nr_budynku"].
    
        Returns:
        pd.DataFrame: DataFrame containing a new column "fuul_address" with processed and formatted addresses.
        """
        try:
            columns = []
            missing_columns = []
            for column in  columns_for_address :
                if column in data.columns:
                    columns.append(column)
                else:
                    missing_columns.append(column)

            if len(columns) <= 1:
                raise KeyError
            elif len(columns) < len(columns_for_address) :
                print(f"To better collect addresses, it is recommended to add the following columns: {missing_columns}")
            
            # Create a copy of columns with data to create an address
            df = data[columns].copy()
        except KeyError as e:
            print(f"No suitable columns found for collecting addresses")
            return None
    
        # Ensure that missing values are replaced with empty strings
        df = df.fillna('')
    
        # Insert the "Country" column with a constant value "Poland"
        df.insert(0, 'Country', 'Poland')
    
        # Create a new column "fuul_address" containing concatenated addresses
        data.loc[:, 'full address'] = df.apply(lambda row: ', '.join(row.astype(str)), axis=1).copy()
    
        # Remove trailing commas and spaces at the beginning and end of each line
        data.loc[:,'full address'] = data['full address'].str.strip(', ')
    
        # Return DataFrame with the new "fuul address" column
        return data



# Clear geodata
class Clear_Data_After_Geocoding:
    """
    This class is designed to filter geocoded data based on the distances between data points and a specified depot location.
    
    Attributes:
    - __teg_filtration_geodata (bool): Flag indicating whether geodata filtration has been performed.
    - depot (dict): Dictionary representing the depot location with keys "address," "latitude," and "longitude."
    - data (pd.DataFrame): DataFrame containing geocoded data.
    - depo (dict): Dictionary representing the depot location used for filtration.
    - __laggards (pd.DataFrame): DataFrame containing data points with distances above a specified threshold.
    - __errors_map: Folium map showing the locations of laggards.
    - map: Folium map showing the locations of filtered data.
    
    Methods:
    - __init__(self, data=None, depo=None, path_data="data/data_after_geocoding/geodata.csv"):
        Initializes the Clear_Data_After_Geocoding object.
    - filtration_geodata(self, threshold=99.97):
        Filters geocoded data based on distances to a depot.
    - get_result(self, revert="all"):
        Retrieves the results of the filtration process.
    - save(self, path="data/data_after_geocoding", file_name=["geodata_clean", "geodata_error"]):
        Saves filtered and laggards data to CSV files.
    - __calculate_distances(self, data):
        Calculates distances to the depot for geocoded data points.
    - __maps(self, data):
        Generates a Folium map showing the locations of data points.
    """

    __teg_filtration_geodata = False
    depot = {
            "address": 'Turystyczna 38, 05-830 Nadarzyn',
            "latitude": 52.105101,
            "longitude": 20.808857
            }
        
    def __init__(self, data = None, depo:dict = None,
                path_data:str = "data/data_after_geocoding/geodata.csv"):
        """
        Initializes the Clear_Data_After_Geocoding object.

        Parameters:
        - data (pd.DataFrame): DataFrame containing geocoded data. If None, data is loaded from the specified CSV file.
        - depo (dict): Dictionary representing the depot location. If None, the default depot is used.
        - path_data (str): Path to the CSV file containing geocoded data.
        """

        self.data = data if data is not None else pd.read_csv(path_data)
        self.depo = depo if depo is not None else self.depot
        self.__laggards = None
        self.__errors_map = None
        self.map = None

    def filtration_geodata( self, threshold:float = 99.97 ):
        """
        Filters geocoded data based on distances to a depot.

        Parameters:
        - threshold (float): Percentile threshold for distance filtering (default is 99.97).
        """
        depo_df = pd.DataFrame([self.depo])
        geo_data = self.data.copy()
        
        # Объединение данных
        geo_data = pd.concat([geo_data, depo_df], ignore_index=True)
    
        # Replace empty strings in the "longitude" column with None
        geo_data.loc[geo_data["longitude"] == "", "longitude"] = None
        # Создать маску для строк с пропущенными значениями в столбце "longitude"
        mask = geo_data["longitude"].isna()
        # Выделить строки с пропущенными значениями в новый DataFrame
        geo_data_nan = geo_data.loc[mask]
        # Оставить только строки без пропущенных значенияй и сбросить индексы
        geo_data = geo_data.loc[~mask].reset_index(drop=True)
    
        # Calculate distances from the centroid to another_days_data
        distances = self.__calculate_distances(geo_data) 
        # Add distances to the DataFrame
        geo_data.loc[:,'distance_to_depo'] = distances

        # Расчет процентиля расстояний
        percentile_distance = np.percentile(geo_data["distance_to_depo"], threshold)
        
        # Фильтрация индексов, где расстояния выше порога
        mask2 = geo_data["distance_to_depo"] >= percentile_distance
        
        # Создание DataFrame с отфильтрованными данными и добавлением депо
        clean_data = geo_data.loc[~mask2, :].reset_index(drop=True)
        laggards_data = geo_data.loc[mask2, :].reset_index(drop=True)

        self.data = clean_data
        laggards_data = pd.concat([laggards_data, geo_data_nan], ignore_index=True)
        self.__laggards = laggards_data

        
        # clean_data = pd.concat([clean_data, depo_df], ignore_index=True)
        # laggards_data = pd.concat([depo_df, laggards_data, geo_data_nan], ignore_index=True)

        self.map = self.__maps(clean_data, depo_df)
        self.__errors_map = self.__maps(laggards_data, depo_df)

        self.__teg_filtration_geodata = True

    def get_result(self, revert = "all"):
        """
        Retrieves the results of the filtration process.

        Parameters:
        - revert (str): Specifies the type of result to retrieve. Options are "all" (default), "errors", or "clean".

        Returns:
        - tuple or pd.DataFrame: Depending on the specified revert option, returns a tuple containing maps and DataFrames or a single DataFrame.
        """
        if not self.__teg_filtration_geodata:
            self.filtration_geodata()
            
        if revert == "all":
            return self.map, self.__errors_map, self.data, self.__laggards
        elif revert == "errors":
            return self.__errors_map, self.__laggards
        elif revert == "clean":
            return self.map, self.data

    def save(self, path="data/data_after_geocoding", file_name=["geodata_clean", "geodata_error"]):
        """
        Saves filtered and laggards data to CSV files.

        Parameters:
        - path (str): Path to the folder where CSV files will be saved (default is "data/data_after_geocoding").
        - file_name (list): List containing two file names for the filtered and laggards data CSV files.

        Returns:
        None
        """
        if not self.__teg_filtration_geodata:
            self.filtration_geodata()
            
        data = self.data.drop("distance_to_depo", axis = 1)
        laggards = self.data.drop("distance_to_depo", axis = 1)
        
        data.to_csv(f"{path}/{file_name[0]}.csv", index=False)
        laggards.to_csv(f"{path}/{file_name[1]}.csv", index=False)
    
        print(f"Data saved to folder: {path}")

    def __calculate_distances(self, data):
        """
        Calculates distances to the depot for geocoded data points.

        Parameters:
        - data (pd.DataFrame): DataFrame containing geocoded data.

        Returns:
        - np.ndarray: Array containing distances to the depot for each data point.
        """
        return np.abs(data['longitude'] - self.depo['longitude']) + np.abs(data['latitude'] - self.depo['latitude'])

    def __maps(self, data, depo_df):
        """
        Generates a Folium map showing the locations of data points.

        Parameters:
        - data (pd.DataFrame): DataFrame containing geocoded data.

        Returns:
        - folium.Map: Folium map object.
        """
        data = pd.concat([data, depo_df], ignore_index=True)
        plot_data = data.dropna(subset = ["longitude", "latitude"])
        m = colored_marker_map(plot_data, "address")

        return m



# Distance
class DistanceMatrixCalculator:

    """
    The DistanceMatrixCalculator class provides methods to calculate distance matrices based on geographic coordinates.

    Parameters:
    - `round_up` (int): Value for rounding the Manhattan distances. Default is 100,000.

    Attributes:
    - `round_up` (int): Value for rounding the Manhattan distances.
    - `data` (pd.DataFrame): DataFrame containing geographic data.

    Methods:
    - `create_geodesic_distance_matrix(data: pd.DataFrame) -> List[List[int]]`:
        Creates a geodesic distance matrix between points based on geographic data.
    - `create_manhattan_distance_matrix(data: pd.DataFrame) -> numpy.ndarray`:
        Creates a Manhattan distance matrix based on geographic coordinates in the data.

    Private Methods:
    - `__check_data() -> pd.DataFrame`:
        Performs checks and updates the geo_data DataFrame.

    Example Usage:
    ```python
    # Create a DistanceMatrixCalculator instance
    calculator = DistanceMatrixCalculator(round_up=100_000)

    # Load geographic data into a DataFrame
    data = pd.DataFrame({"latitude": [40.7128, 37.7749], "longitude": [-74.0060, -122.4194]})

    # Calculate geodesic distance matrix
    geodesic_matrix = calculator.create_geodesic_distance_matrix(data)
    print(geodesic_matrix)

    # Calculate Manhattan distance matrix
    manhattan_matrix = calculator.create_manhattan_distance_matrix(data)
    print(manhattan_matrix)
    ```
    """
    
    def __init__(self, round_up=100_000):
        """
        Initializes the DistanceMatrixCalculator class.

        Args:
            round_up (int): Value for rounding the Manhattan distances. Default is 10,000.

        Example:
        >>> import pandas as pd
        >>> data = pd.DataFrame({"latitude": [40.7128, 37.7749], "longitude": [-74.0060, -122.4194]})
        >>> calculator = DistanceMatrixCalculator()
        >>> calculator.data = data
        """
        self.round_up = round_up
        self.data = pd.DataFrame()

    def create_geodesic_distance_matrix(self, data):
        """
        Creates a geodesic distance matrix between points based on geographic data.

        Args:
            data (pd.DataFrame): DataFrame containing geographic data in the columns "latitude" and "longitude".

        Returns:
            List[List[int]]: Distance matrix between points.

        Note:
        - The input data 'data' should contain the columns 'latitude' and 'longitude'.
        """
        self.data = data
        self.__check_data()

        self.data[["latitude", "longitude"]] = self.data[["latitude", "longitude"]].astype(float)
        num_points = len(self.data)
        distance_matrix = [[0 for _ in range(num_points)] for _ in range(num_points)]

        for i in range(num_points):
            for j in range(i + 1, num_points):
                if j < num_points:
                    coord1 = (self.data.at[i, "latitude"], self.data.at[i, "longitude"])
                    coord2 = (self.data.at[j, "latitude"], self.data.at[j, "longitude"])
                    distance = geodesic(coord1, coord2).kilometers
                    distance_matrix[i][j] = int(distance)
                    distance_matrix[j][i] = int(distance)

        return distance_matrix

    def create_manhattan_distance_matrix(self, data):
        """
        Creates a Manhattan distance matrix based on geographic coordinates in the data.

        Args:
            data (pd.DataFrame): DataFrame containing geographic data in the columns "longitude" and "latitude".

        Returns:
            numpy.ndarray: Manhattan distance matrix.

        Note:
        - The input data 'data' should contain the columns 'longitude' and 'latitude'.
        """
        self.data = data
        self.__check_data()

        coords = self.data[["longitude", "latitude"]].values
        num_points = len(coords)
        distance_matrix = np.zeros((num_points, num_points))

        for i in range(num_points):
            for j in range(num_points):
                lon1, lat1 = coords[i]
                lon2, lat2 = coords[j]
                distance = abs(lon1 - lon2) + abs(lat1 - lat2)
                distance_matrix[i][j] = distance

        return (distance_matrix * self.round_up).astype(int)
        
            
    def __check_data(self):
        """
        Performs checks and updates the geo_data DataFrame.

        Returns:
            pd.DataFrame: Updated geo_data DataFrame.
        """
        
        self.data = self.data.reset_index(drop=True)
        self.data.replace('', np.nan, inplace=True)
        self.data.dropna(subset=["longitude", "latitude"], inplace=True)

        
        
        self.data["longitude"] = self.data["longitude"].astype("float32")
        self.data["latitude"] = self.data["latitude"].astype("float32")

        return self.data

        













