from lib.imports import *
from project_functions.tsp_solver import *


# # merge KnapsackSolver and ScheduleFiller 
class FillingEmptyTransport(OrToolsRoutingProblemSolver, GeneticAlgorithmRoutingProblemSolver):
    """
        A class that represents a solution for filling empty transport routes in a capacitated vehicle routing problem (CVRP).
        
        Parameters:
        - data (pd.DataFrame): DataFrame containing geographical and routing data.
        - depo (str|list[int]): Address or list of indices representing the depot location. Defaults to "Zawodzie 18, 02-981 Warszawa".
        - column (str): Name of the column containing grouping information. Defaults to "groups".
        - capacity (int): Maximum capacity of the vehicles. Defaults to 21,000.
        - acceptable_shortfalls (float): Acceptable percentage of capacity shortfall. Defaults to 0.1.
        - acceptable_excess (float): Acceptable percentage of capacity excess. Defaults to 0.1.
        - verbose (bool): If True, print additional information during optimization. Defaults to False.
        - solver_type (str): Type of solver to be used, either "or_tools" or "genetic". Defaults to "or_tools".
    
        Attributes:
        - data (pd.DataFrame): DataFrame containing the original routing data.
        - depo (str|list[int]): Address or indices representing the depot location.
        - solver_type (str): Type of solver used for optimization.
        - column (str): Name of the column containing grouping information.
        - capacity (int): Maximum capacity of the vehicles.
        - min_capacity (int): Minimum acceptable capacity after considering shortfalls.
        - max_capacity (int): Maximum acceptable capacity after considering excess.
        - acceptable_shortfalls (float): Acceptable percentage of capacity shortfall.
        - acceptable_excess (float): Acceptable percentage of capacity excess.
        - verbose (bool): If True, print additional information during optimization.
    
        Methods:
        - _knapsack_solver(data=None, column=None): Solves the capacitated vehicle routing problem (CVRP) using the knapsack approach.
        - get_knapsack_solver(): Retrieves the solution of the knapsack solver.
        - solver_filling(): Launches the schedule optimization and returns the optimized schedule data.
        - __fill_data_capacity(): Fills data in the schedule taking into account the maximum capacity.
        - __process_day(): Processes data by frequency, waste code, and day of the week.
        - __preprocessing(data): Performs checks and updates the DataFrame.
        - __calculate_centroid(data): Calculates the centroid of the given data.
        - __calculate_distances(data, centroid): Calculates distances to the centroid for the given data.
    """
    optimization_for_the_route = []
    __tag_solver_filling = False
    
    # List to sort the days_week list
    order_dict = {'Pn': 1, 'Wt': 2, 'Sr': 3, 'Cz': 4, 'Pt': 5, 'So': 6}
    
    def __init__(self, data: pd.DataFrame, 
                 depo: str|list[int] = "Zawodzie 18, 02-981 Warszawa", # None
                 column: str ="groups", 
                 capacity: int=21_000, 
                 acceptable_shortfalls: float = 0.1,
                 acceptable_excess: float = 0.1,
                 verbose: bool=False,
                 solver_type: str = "or_tools"):
            

        # Initialize attributes
        self.data = data.copy() 
        self.depo = depo
        self.solver_type = solver_type
        self.column= column
        self.capacity = capacity
        
        self.min_capacity = capacity - (capacity * acceptable_shortfalls) 
        self.max_capacity = capacity - (capacity * acceptable_excess) 
        
        self.acceptable_shortfalls = acceptable_shortfalls
        self.acceptable_excess = acceptable_excess
        self.verbose = verbose
        


    def _knapsack_solver(self, data=None, column= None):
        """
        Solves the capacitated vehicle routing problem (CVRP) using the knapsack approach.
    
        Args:
            data (pd.DataFrame, optional): DataFrame containing geographical and routing data. Defaults to None.
            column (str, optional): Name of the column containing grouping information. Defaults to "groups".
    
        Returns:
            pd.DataFrame: DataFrame with additional information about outliers marked in the "outliner" column.
        """
        
        # If data is not provided, use data from the OrToolsRoutingProblemSolver object
        if not isinstance(data, pd.DataFrame):
            data = self.data.copy()

        if not isinstance(column, str):
            column = self.column

        # Create the "outliner" column and initialize it with 0
        data.loc[:, "outliner"] = 0
    
        # Get unique groups
        groups = data[column].unique() 
    
        # Lists to store route data and indices
        all_routes_data = []

        # Iterate over groups
        for group in tqdm(groups, desc='route planning'):
            if self.solver_type == "genetic":
                # Create an OrToolsRoutingProblemSolver instance for each group
                instance = GeneticAlgorithmRoutingProblemSolver(geo_data=data[data[column] == group],
                                                              vehicle_capacities=self.capacity, 
                                                              num_vehicles=ceil(sum(data["Pojemnosc"]) / self.capacity))

                # Solve the routing problem
                route_data = instance.genetic_route_solver(verbose=False, return_coordinates=False)
                
            elif self.solver_type == "or_tools":
                # Create an OrToolsRoutingProblemSolver instance for each group
                instance = OrToolsRoutingProblemSolver(geo_data=data[data[column] == group],
                                                      vehicle_capacities=self.capacity, 
                                                      num_vehicles=ceil(sum(data["Pojemnosc"]) / self.capacity))
        
                # Solve the routing problem
                route_data = instance.ort_route_solver(return_coordinates=False)
                
            else:
                raise TypeError("Wrong router type. Please select one of the available routers: genetic|or_tools")
    
            # Get unique routes, excluding NaN
            routs = route_data["route"].unique()
            routs = routs[~np.isnan(routs)]
            
            # Iterate over routes
            route_data_list  = []
            for rout in routs:
                route_df = route_data.loc[route_data["route"] == rout]
                route_df = route_df.iloc[1:-1]
                
                # Calculate the total capacity of the route
                capacity_rout = route_df[ "Pojemnosc"].sum()
                
                # If the capacity of the route is less than the minimum, set outliner to 1
                if capacity_rout < self.min_capacity:
                    route_df["outliner"] = 1
                route_data_list .append(route_df)
            route_data = pd.concat(route_data_list ).drop("route", axis =1)
            
            # Add route data to the list
            all_routes_data.append(route_data)
            
        # Concatenate all route data into one DataFrame
        data = pd.concat(all_routes_data).reset_index(drop = True)

        return  data

    def get_knapsack_solver(self):
        """
        Solves the capacitated vehicle routing problem (CVRP) using the knapsack approach.
    
        Args:
            data (pd.DataFrame, optional): DataFrame containing geographical and routing data. Defaults to None.
            column (str, optional): Name of the column containing grouping information. Defaults to "groups".
    
        Returns:
            pd.DataFrame: DataFrame with additional information about outliers marked in the "outliner" column.
        """
        if self.__tag_solver_filling: 
            data = self.data.copy()
            data = data.drop(["distance_to_centroid","new_frequency"], axis = 1)
            return data
        else:
            data = self.solver_filling()
            data = data.drop(["distance_to_centroid","new_frequency"], axis = 1)
            return data

    def solver_filling(self):
        """
        The main method that launches the schedule optimization.
    
        Returns:
        - DataFrame: Optimized schedule data.
        """
        
        self.data, self.df = self.__preprocessing( self.data ) if "outliner" in self.data.columns\
            else  self.__preprocessing(self._knapsack_solver(data = self.data))
        
        try:
            # Create a list to store new data
            new_schedules_list = []
        
            # Run optimization for each schedule
            for new_schedule in tqdm(self.__fill_data_capacity()):
                if new_schedule is not None:
                    # Create a DataFrame with sorted data 
                    if isinstance(new_schedule, pd.DataFrame):
                        new_schedules_list.append(new_schedule)
                    else:
                        # If the object is not a DataFrame, raise a TypeError
                        raise TypeError(f"Method solver_filling skipping non-DataFrame object: {new_schedule}")
                else: 
                    # If new_schedule is None, raise a ValueError as there is insufficient data for optimization
                    raise ValueError("No data available for optimization. Check input data or adjust optimization parameters.")

                    
            try:
                # Concatenate DataFrames from the list
                new_schedules = pd.concat(new_schedules_list, ignore_index=False)
            except ValueError as e:
                # If there is too little data to process frequency, raise a ValueError
                error_message = "Too little data to process frequency"
                raise ValueError(error_message)
        
            # Select indices 
            index_new_schedules = new_schedules.index
            
            # Update values in new_frequency
            self.data.loc[index_new_schedules, "new_frequency"] = new_schedules["new_frequency"]
          
            # Drop unnecessary columns 
            self.data.drop("distance_to_centroid", axis=1, inplace=True)
            
            # Fill missing values with old data 
            self.data['new_frequency'].fillna(self.data['groups'], inplace=True)

            # raise the flag so that the solver does not repeat itself when trying to retrieve 
            # data via the get_knapsack_solver function
            self.__tag_solver_filling = True
            
            return self.data
            
        except TypeError as e:
            # Print a more informative error message
            error_message = f"TypeError occurred: {str(e)}"
            # Re-raise the TypeError with the enhanced error message
            raise TypeError(error_message)

    def __fill_data_capacity(self):
        """
        Fills data in the schedule taking into account the maximum capacity.
    
        :param data_for_filling: Data to be filled
        :param data: Original data
        :return: Updated schedule
        """
        for day_data, data_for_additional_filling, route_capacity in self.__process_day():
            if day_data.iloc[0]["groups"] not in self.optimization_for_the_route:
                self.optimization_for_the_route.append(day_data.iloc[0]["groups"])
            
            # If day_data is empty, continue
            if day_data.empty or data_for_additional_filling.empty:
                # if not day_data.empty:
                #     yield day_data
                continue
                
            # Calculate the current sum of capacity in day_data
            cap = day_data["Pojemnosc"].sum()
            
            # Create a list to store new data
            new_rows = []
            return_new_rows = True
        
            # Iterate through rows of data_for_additional_filling to fill in day_data
            for i, r in data_for_additional_filling.iterrows():
                # Check if groups is not in optimization_for_the_route
                if r["groups"] not in self.optimization_for_the_route:                    
                    # Check how many routes per week are made to this address
                    if len(r["Dni tygodnia"].split(",")) == 1:
                        # Check if adding another row will not exceed the capacity
                        if (cap + r["Pojemnosc"]) < route_capacity:
                            # Update the current sum of capacity
                            cap += r["Pojemnosc"]
                            # Create a new row in the form of a DataFrame
                            new_row = pd.DataFrame(r).T
                            # Add the new row to the list
                            new_rows.append(new_row)
                        # If the capacity exceeds the threshold value, then
                        else:
                            # If there are new rows 
                            if new_rows:
                                # Concatenate new data (new_rows) to day_data 
                                # Create a new DataFrame from the list of new rows
                                new_rows_df = pd.concat(new_rows, ignore_index=False)
                                # Fill "new_label" with a value from day_data
                                new_rows_df["new_frequency"] = day_data.iloc[0]["groups"]
                                # Merge the new DataFrame with day_data
                                day_data = pd.concat([day_data, new_rows_df], ignore_index=False) # , ignore_index=True
                                # Return the filled day_data
                                yield day_data
                                return_new_rows = False
                                break
                    # If the same address is used more than once in one week, skip this row 
                    else: 
                        continue
                # If groups is in optimization_for_the_route, skip this row
                else:
                    # print("отработало условие if r[groups] not in self.optimization_for_the_route" )
                    continue
                    
            # If there are new rows, but there was not enough weight to add them to day_data, then intercept and add the data 
            if return_new_rows:
                if new_rows:
                    # Concatenate new data (new_rows) to day_data 
                    # Create a new DataFrame from the list of new rows
                    new_rows_df = pd.concat(new_rows, ignore_index=False)
                    # Fill "new_label" with a value from day_data
                    new_rows_df["new_frequency"] = day_data.iloc[0]["groups"]
                    # Check if new_rows_df contains at least one row
                    if not new_rows_df.empty:
                        # Merge the new DataFrame with day_data
                        day_data = pd.concat([day_data, new_rows_df], ignore_index=False)
                        # Return the filled day_data
                        yield day_data

    def __process_frequency(self):
        """
        Selects data by frequency.

        :param frequencies: Unique frequencies
        :param end: End index for processing
        :param waste_codes: Unique waste codes
        """
        # Get unique frequencies from the 'Częstotliwość' column, excluding non-string values
        frequencies = sorted([i for i in self.data["Częstotliwość"].unique() if isinstance(i, str)])
        
        # Dictionary to store frequencies grouped by their first character
        frequencies_dictionary = {}
        
        # Iterate through frequencies and group them by their first character
        for frequency in frequencies:
            key = frequency[0]
            
            if key not in frequencies_dictionary:
                frequencies_dictionary[key] = [frequency]  
            else:
                frequencies_dictionary[key].append(frequency)
        
        # Iterate through grouped frequencies and yield filtered rows for each group
        for frequency in frequencies_dictionary.values():        
            # Filter rows based on the 'Częstotliwość' column
            frequency_data = self.data[self.data["Częstotliwość"].isin(frequency)]
            # Yield the filtered rows for further processing
            yield frequency_data

    def __process_waste_code(self):
        """
        Processes data by waste code.

        :param filtered_rows: Filtered rows
        :param waste_code: Waste code
        """
        for df in self.__process_frequency():
            waste_codes = df["Kod odpadu"].unique()
            for waste_code in waste_codes:
                waste_code_data = df[df["Kod odpadu"] == waste_code].copy()
                yield waste_code_data

    def __process_day(self):
        """
        Processes data by frequency, waste code, and day of the week.
        """
        for waste_code_data in self.__process_waste_code():

            # Select unique days with Częstotliwość
            days_week = [i for i in waste_code_data['Częstotliwość'].unique() if i[1:] in self.order_dict.keys()]
            
            # Sort the days_week list
            days_week = sorted(days_week, key=lambda x: self.order_dict[x[1:]] )
            
            for day in days_week:
                
                day_data = waste_code_data[waste_code_data["Częstotliwość"] == day]
                another_days_data = waste_code_data[waste_code_data["Częstotliwość"] != day]

                if not day_data.empty:
                    
                    # Calculate capacity by summing the values in the "Pojemnosc" column in the DataFrame day_data
                    capacity = day_data["Pojemnosc"].sum()
                    # Determine the required number of routes by rounding up the ratio of the total capacity to the given capacity on the route
                    routes = ceil(capacity / self.capacity)
                    # Calculate the minimum capacity for all routes by multiplying the number of routes by the minimum capacity on the route
                    min_route_capacity = routes * self.min_capacity
                    # Calculate the total capacity for all routes by multiplying the number of routes by the given capacity on the route
                    route_capacity = routes * self.capacity 
                    
                    if day_data.iloc[0]["groups"] not in self.optimization_for_the_route:
                        self.optimization_for_the_route.append(day_data.iloc[0]["groups"])
                        
                    if  day_data["Pojemnosc"].sum() >= min_route_capacity:
                        continue
                        
        
                    # Calculate the centroid
                    centroid = self.__calculate_centroid(day_data)
                    
                    # Calculate distances from the centroid to other points in the group
                    intra_group_distance = self.__calculate_distances(day_data, centroid)    
                    
                    # Set the threshold value 
                    threshold = intra_group_distance.median()
                    
                    # Calculate distances from the centroid to another_days_data
                    distances = self.__calculate_distances(another_days_data, centroid) 
                    
                    # Add distances to the DataFrame
                    another_days_data.loc[:,'distance_to_centroid'] = distances
                    
    
                    yield day_data, another_days_data.sort_values("distance_to_centroid"), route_capacity
                
                else :
                    continue

    def __preprocessing(self, data):
        """
        Performs checks and updates the geo_data DataFrame.

        Returns:
        pd.DataFrame: Updated geo_data DataFrame.
        """
        
        # Create a new column (distance_to_centroid) to store the distance 
        data.loc[:, 'distance_to_centroid'] = 0
        # Create a column to store the new schedule 
        data.loc[:, "new_frequency"] = None

        # Drop depo from data
        if isinstance(self.depo, str):
            data = data[data["full address"] != self.depo ].reset_index(drop = True)
        elif isinstance(self.depo, list):
            data = data.drop(self.depo)
            
        # Split the data 
        df = data[data["outliner"] == 1].copy()

        return data, df

    def __calculate_centroid(self, data):
        """
        Calculates the centroid.

        :param day_data: Data for the day of the week
        :return: Centroid
        """
        return data[['longitude', 'latitude']].mean()
 
    def __calculate_distances(self, data, centroid):
        """
        Calculates distances to the centroid for other days.

        :param another_days_data: Data for other days of the week
        :param centroid: Centroid
        :return: Distances to the centroid
        """
        return np.abs(data['longitude'] - centroid['longitude']) + np.abs(data['latitude'] - centroid['latitude'])



## Supporting Functions
def check_frequency_difference(routs):
    """
    Check for missing data in the 'new_schedules' DataFrame.

    Parameters:
    - new_schedules: DataFrame - The input DataFrame containing new schedules information.

    Output:
    - Prints information about missing data in the 'groups' and 'new_frequency' columns.
      If no missing data is found, it prints "no missing data."
    """
    # Получить уникальные значения из столбца 'new_frequency'
    new_frequency = routs['new_frequency'].unique()
    new_frequency = new_frequency[pd.notna(new_frequency)]
    
    # Получить уникальные значения из столбца 'groups'
    frequency_groups = routs['groups'].unique()
    frequency_groups = frequency_groups[pd.notna(frequency_groups)]

    missing = False
    count = 0

    # Проверка на отсутствие значений в 'new_frequency', которые отсутствуют в 'groups'
    missing_in_groups = np.setdiff1d(new_frequency, frequency_groups, assume_unique=False)
    missing_in_groups = missing_in_groups[~pd.isna(missing_in_groups)]
    if len(missing_in_groups) > 0:
        print("Values not in groups:", missing_in_groups)
        count += len(missing_in_groups)
        missing = False

    # Проверка на отсутствие значений в 'groups', которые отсутствуют в 'new_frequency'
    missing_in_new_frequency = np.setdiff1d(frequency_groups, new_frequency, assume_unique=False)
    missing_in_new_frequency = missing_in_new_frequency[~pd.isna(missing_in_new_frequency)]
    if len(missing_in_new_frequency) > 0:
        print("Values not in new_frequency:", missing_in_new_frequency)
        count += len(missing_in_new_frequency)
        missing = False

    # Печать результата
    if missing or not count:
        print("No missing data")

















