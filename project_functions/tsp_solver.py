from lib.project_functions import *
from project_functions.visualization import RouteInfoAndMap

# or tools Solver
class OrToolsRoutingProblemSolver(DistanceMatrixCalculator, Geocoder ):
    """
    A class for solving routing problems using the OR-Tools library.

    Attributes:
        depo (pd.DataFrame): DataFrame containing the depot information.
        geo_data (pd.DataFrame): DataFrame containing geographical data for the problem.
        function_distance_matrix (callable): A callable function for computing the distance matrix.
        num_vehicles (int): The number of vehicles available.
        depot (int): Index of the depot in the geo_data DataFrame.
        vehicle_capacities (list): List of vehicle capacities. Default is None.
        time_limit_seconds (int): Time limit in seconds for the optimization process. Default is 120.
        max_distance (int): Maximum distance allowed for the routes. Default is 1,000,000.
        data (dict): Dictionary to store various data during the optimization process.
        manager: A routing manager for handling indices.
        routing: A routing model to define the problem.
        solution: A solution obtained from solving the routing problem.

    Methods:
        __init__: Initializes the OrToolsRoutingProblemSolver object.
        check_data: Performs checks and updates the geo_data DataFrame.
        create_data_model: Creates the data model for solving the routing problem.
        solve_routing_problem: Solves the routing problem using the data model and prints the solution.
        route_information: Prints the solution information or an error message.
        get_solution: Retrieves the solution and stores it in a list, appending an error message if no solution is found.
        generate_route_coordinates: Generates the coordinates of the routes based on the obtained solution.
        __distance_callback: Computes the distance between two points using the distance matrix.
        __demand_callback: Retrieves the demand at a specific index.

    Example:
        solver = OrToolsRoutingProblemSolver(geo_data, distance_matrix_function)
        solver.solve_routing_problem()
        solver.route_information(verbose=True)
    """

    address_depo = pd.DataFrame({"full address": ['Zawodzie 18, 02-981 Warszawa'],
                                "latitude": [52.183425], "longitude": [21.085442],
                                "Pojemnosc": 0 
                                   })

    def __init__(self, geo_data: pd.DataFrame, 
                 num_vehicles: int = None,
                 vehicle_capacities: list = None, 
                 time_limit_seconds: int = 120,
                 max_distance: int = 1_000_000,
                 depo = None):
        """
        Initializes the OrToolsRoutingProblemSolver object.

        Args:
            geo_data (pd.DataFrame): DataFrame containing geographical data for the problem.
            function_distance_matrix (callable): A callable function for computing the distance matrix.
            num_vehicles (int, optional): The number of vehicles available. Defaults to 1.
            vehicle_capacities (list, optional): List of vehicle capacities. Defaults to None.
            time_limit_seconds (int, optional): Time limit in seconds for the optimization process. Defaults to 120.
            max_distance (int, optional): Maximum distance allowed for the routes. Defaults to 1,000,000.
        """
        DistanceMatrixCalculator.__init__(self, round_up=100_000)
        Geocoder.__init__(self, max_threads=10)
        self.depo = depo if depo else self.address_depo
        self.geo_data = geo_data.copy() 
        self.vehicle_capacities = vehicle_capacities if vehicle_capacities else 21_000
        self.num_vehicles = num_vehicles if num_vehicles else ceil( sum(geo_data["Pojemnosc"]) / self.vehicle_capacities)
        self.depot = 0
        self.time_limit_seconds = time_limit_seconds
        self.max_distance = max_distance
        self.data = {}
        self.manager = None
        self.routing = None
        self.solution = None
        self.geo_data = self.__check_data() 
        self.distance_matrix = self.create_manhattan_distance_matrix(self.geo_data)

    def ort_route_information(self, verbose=True):
        """
        Prints the solution if it exists; otherwise, it prints an error message.
    
        Args:
            verbose (bool, optional): If True, prints detailed information. Defaults to True.
    
        Returns:
            tuple: A tuple containing total_distance, total_load, and time.
                - total_distance (float): Total distance of all routes in kilometers.
                - total_load (float): Total load of all routes.
                - time (typle): Total duration of all routes in hours and minut.
    
        Notes:
            - This method assumes the existence of certain attributes and objects like self.solution,
              self.routing, self.manager, self.geo_data, and self.depot.
            - The duration calculation assumes a speed of 30 km/h.
    
        Example:
            instance = YourClass()
            instance.route_information(verbose=True)
        """
        if self.solution:
            
            def convert_length_to_duration(length, speed, string=True):
                """
                Converts length to a time interval.
        
                Args:
                    length (float): Length to convert.
                    speed (float): Speed for the conversion.
                    string (bool, optional): If True, returns a string in the format 'hours:minutes'.
                        If False, returns a tuple of hours and minutes. Defaults to True.
        
                Returns:
                    str or tuple: Converted duration in hours:minutes format if string is True,
                                  otherwise, a tuple of hours and minutes.
                """
                # Convert length to time interval
                duration_hours = length / speed
                duration_hours = duration_hours + ((len(self.geo_data) * 2.2 ) / 60 )
                
                # Separate hours and minutes
                hours = int(duration_hours)
                minutes = int((duration_hours - hours) * 60)
                
                if string : return f'{hours}:{minutes} '
                else: return hours, minutes
    
            print(f"Objective: {self.solution.ObjectiveValue()}")
            total_distance = 0
            total_load = 0
            for vehicle_id in range(self.data["num_vehicles"]):
                index = self.routing.Start(vehicle_id)
                plan_output = f"Route for vehicle {vehicle_id}:\n"
                route_distance = 0
                route_load = 0
                while not self.routing.IsEnd(index):
                    node_index = self.manager.IndexToNode(index)
                    route_load += self.data["demands"][node_index] 
                    plan_output += f" {node_index} Load({route_load}) -> "
                    previous_index = index
                    index = self.solution.Value(self.routing.NextVar(index))
                    route_distance += self.routing.GetArcCostForVehicle( previous_index, index, vehicle_id ) / 1000
                
                plan_output += f" {self.manager.IndexToNode(index)} Load({route_load})\n"
                plan_output += f"Distance of the route: {route_distance} km\n"
                plan_output += f'Route time {convert_length_to_duration(route_distance, 30)} h/m\n'
                plan_output += f"Load of the route: {route_load}\n"
                print()
                print(plan_output)
                print()
                
                total_distance += route_distance
                total_load += route_load / 1000
                
            total_time = convert_length_to_duration(total_distance, speed=33, string=False)
            print(f"Approximate distance of all routes: {total_distance} km")
            print(f"Total load of all routes: {total_load} T")
            print(f'Approximate duration of route {total_time} h/m')
            print()
            
            return total_distance, total_load, total_time
        else:
            print()
            print('No solution found!')
            return 0, 0, (0, 0)

    def ort_route_solver (self, verbose=False, return_coordinates=False): 
        """
        Generates route data based on the obtained solution and adds a 'route' column to the DataFrame.
    
        Parameters:
        - verbose (bool, optional): If True, print verbose information during the process.
        - return_index (bool, optional): If True, return both the DataFrame and the route indices.
    
        Returns:
        - pd.DataFrame or tuple: If return_index is False, returns the DataFrame with added 'route' column. 
          If return_index is True, returns a tuple containing the DataFrame and the route indices.
    
        Example:
        >>> geo_data = generate_route_data(verbose=True, return_index=True)
        """
    
        # Obtain route solutions using the get_solution method
        solutions = self.__get_solution(verbose=verbose)
        
        # Initialize the 'route' column with zeros
        self.geo_data.loc[:, "route"] = 0
    
        # Check if a valid solution exists
        if solutions[0] != "Brak rozwiązania!":
            
            all_routs = []
            count = 1
            
            if not return_coordinates:
                for solution in solutions:
                    
                    # Extract latitude and longitude coordinates for each index in the route
                    df = pd.DataFrame([self.geo_data.iloc[i] for i in solution])
                    df = df.reset_index(drop = True)
                    df = df.assign(route=count)
                    
                    all_routs.append(df)
                    count += 1
                    
                data = pd.concat(all_routs).reset_index(names= "index").sort_values(["route", "index"])
                data = data.drop("index", axis = 1)
                
                return data

            else :
                for solution in solutions:
                    # Extract latitude and longitude coordinates for each index in the route
                    df = pd.DataFrame()
                    
                    df["latitude"] = [self.geo_data.loc[solution[i], "latitude"] for i in range(len(solution))]
                    df["longitude"] = [self.geo_data.loc[solution[i], "longitude"] for i in range(len(solution))]
                    df = df.assign(route=count)  
                    
                    all_routs.append(df)
                    count += 1
                    
                return all_routs
    
    def __check_data(self) -> pd.DataFrame:
        """
        Performs checks and updates the geo_data DataFrame.

        Returns:
        pd.DataFrame: Updated geo_data DataFrame.
        """
        data = self.geo_data
        columns = data.columns
        if "longitude" not in columns and "latitude" not in columns :
            data = self._geocoding()

        if isinstance(self.depo, dict):
            self.depo = pd.DataFrame([self.depo])
            
        # Concatenating the depot information with the geo_data DataFrame
        if data["full address"].ne(self.depo["full address"]).any():
            data = pd.concat([self.depo, data], axis=0, ignore_index=True).reset_index(drop = True)

        
        # Converting longitude and latitude to float
        data.loc[:, "longitude"] = data["longitude"].astype("float32")
        data.loc[:, "latitude"] = data["latitude"].astype("float32")
        
        return data

    def _geocoding(self):
        """
        Performs geocoding of addresses and prepares data for the knapsack problem.

        Returns:
            DataFrame: Processed data with geocoded information.
        """
        print("geocoding process")

        data_for_geocoding = self.address_collector(self.geo_data.copy())
        data_for_geocoding.reset_index(drop = True, inplace = True)
        
        unique_values = data_for_geocoding["full address"].drop_duplicates()
        unique_indices = unique_values.index.to_numpy()

        unique_data_for_geocoding = data_for_geocoding.iloc[unique_indices]
                
        self.geocode_addresses(unique_data_for_geocoding)
        self.save_results_to_csv()

        # Get geography data and merge it with the source data
        result_coding = self.get_results()

        result_coding = pd.DataFrame(result_coding)
        
        merge_data = data_for_geocoding.merge(result_coding, how="inner", left_on="full address", right_on="address").drop("address", axis = 1)
        

        # Prepare data for the task
        columns_to_replace = ["Pojemnosc", "longitude", "latitude"]
        merge_data[columns_to_replace] = merge_data[columns_to_replace].replace('', np.nan)

        merge_data = merge_data.dropna(subset=["longitude", "latitude"])
        merge_data = merge_data.reset_index(drop=True)

        # Display information about the number of found and missing data
        missing = len(data_for_geocoding) - len(merge_data)
        if missing:
            print(f"{missing} missing data found in the data.")
            # Display the addresses that were not successfully geocoded, sorted by "full address"
            missing_addresses = data_for_geocoding.loc[~data_for_geocoding["full address"].isin(merge_data["full address"]), "full address"]
            missing_addresses_sorted = missing_addresses.sort_values()
            print("Addresses missing data:")
            display(missing_addresses_sorted)
        else:
            print("All data successfully geocoded.")

        return merge_data

    def __create_data_model(self):
        """
        Creates the data model for solving the routing problem.

        Returns:
        dict: A dictionary containing the required data for the optimization process.
        """
        self.__check_data()
        
        self.data = {}
        self.data['distance_matrix'] = self.distance_matrix
        self.data["demands"] = self.geo_data["Pojemnosc"].astype("int")
        self.data["vehicle_capacities"] = [self.vehicle_capacities if self.vehicle_capacities else 20_000 for _ in range(self.num_vehicles)]
        self.data['num_vehicles'] = self.num_vehicles
        self.data['depot'] = self.depot
        
        return self.data

    def __solve_routing_problem(self) -> tuple:
        """
        Solves a routing problem using a data model.

        Returns:
        tuple: A tuple containing the routing index manager, the routing model, and the solution.
        """
        # Creating a routing index manager with the specified parameters
        self.manager = pywrapcp.RoutingIndexManager(len(self.data['distance_matrix']), self.data['num_vehicles'], self.data['depot'])
        # Creating a routing model
        self.routing = pywrapcp.RoutingModel(self.manager)

        # Registering the transit callback function
        transit_callback_index = self.routing.RegisterTransitCallback(self.__distance_callback)
        self.routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Defining the distance dimension 
        dimension_name = "Distance"
        self.routing.AddDimension(
            transit_callback_index,
            0,
            self.max_distance,
            True,
            dimension_name,
        )

        distance_dimension = self.routing.GetDimensionOrDie(dimension_name)
        distance_dimension.SetGlobalSpanCostCoefficient(100)

        # Registering the demand callback function 
        demand_callback_index = self.routing.RegisterUnaryTransitCallback(self.__demand_callback)
        self.routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,
            self.data["vehicle_capacities"],
            True,
            "Capacity",
        )

        # Setting the search parameters for the routing problem
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.time_limit.seconds = self.time_limit_seconds
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
        search_parameters.log_search = True

        # Solving the routing problem with the specified parameters
        self.solution = self.routing.SolveWithParameters(search_parameters)

        return self.manager, self.routing, self.solution
  
    def __distance_callback(self, from_index, to_index):
        """
        Computes the distance between two points using the distance matrix.

        Args:
        from_index (int): Index of the starting point.
        to_index (int): Index of the ending point.

        Returns:
        float: The distance between the two points.
        """
        from_node = self.manager.IndexToNode(from_index)
        to_node = self.manager.IndexToNode(to_index)
        return self.data['distance_matrix'][from_node][to_node]

    def __demand_callback(self, from_index):
        """
        Retrieves the demand at a specific index.

        Args:
        from_index (int): Index of the location.

        Returns:
        int: The demand at the specified location.
        """
        from_node = self.manager.IndexToNode(from_index)
        return self.data["demands"][from_node]

    def __get_solution(self, verbose = False):
        """
        Retrieves the solution and stores it in a list, appending an error message if no solution is found.
        If no solution is found, recursively calls itself with an increased number of vehicles by 1.
    
        Args:
        num_vehicles (int): Number of vehicles for the routing problem. Default is None.
    
        Returns:
        list: A list containing the solutions or an error message.
        """
        self.__create_data_model()
        self.__solve_routing_problem()
        if verbose : self.ort_route_information()
    
    
        if self.solution:
            routes = []
            for route_nbr in range(self.routing.vehicles()):
                index = self.routing.Start(route_nbr)
                route = [self.manager.IndexToNode(index)]
                while not self.routing.IsEnd(index):
                  index = self.solution.Value(self.routing.NextVar(index))
                  route.append(self.manager.IndexToNode(index))
                routes.append(route)
            return routes
        else:
            print('No solution found with the current number of vehicles.')
    
            # Recursive call with an increased number of vehicles by 1
            if num_vehicles < 1000:  
                print('Trying again with an increased number of vehicles.')
                return self.__get_solution(num_vehicles + 1)
            else:
                solutions.append('No solution even with increased number of vehicles.')
            return None





### Genetic Algorithm Solver
class GeneticAlgorithmRoutingProblemSolver(DistanceMatrixCalculator, Geocoder):
    """
    GeneticAlgorithmRoutingProblemSolver class is designed to solve routing problems using a genetic algorithm.

    Parameters:
    - data (pd.DataFrame): Input data containing information about addresses, latitude, longitude, and capacity.
    - VOLUME (float): Desired volume for each route.
    - POPULATION_SIZE (int): Size of the population in the genetic algorithm. Default is 500.
    - MAX_GENERATIONS (int): Maximum number of generations in the genetic algorithm. Default is 1000.
    - deviation_percentage_weight (float): Weight factor for volume deviation. Default is 0.05.
    - CROSSOVER (float): Crossover probability in the genetic algorithm. Default is 0.75.
    - P_MUTATION (float): Mutation probability in the genetic algorithm. Default is 0.75.
    - HALL_OF_FAME_SIZE (int): Size of the Hall of Fame in the genetic algorithm. Default is 5.
    - startV (int): Starting point for routes. Default is 0.
    - endV (int or str): Ending point for routes. Default is "startV," indicating the same as the starting point.
    - penalty (float): Penalty factor for weight deviation. Default is 0.2.
    - depo (pd.DataFrame): Depot information containing the starting point coordinates. Default is None.
    - number_of_routes (int): Desired number of routes. Default is calculated based on the total capacity.

    Methods:
    - genetic_route_solver(verbose=True, return_coordinates=False): Solves the routing problem using a genetic algorithm.
    - genetic_route_information(verbose=True): Displays information about the optimized route.
    - __check_data(data): Performs checks and updates the input data.
    - __eaSimpleElitism(population, toolbox, cxpb, mutpb, ngen, stats=None, halloffame=None, verbose=None, callback=None):
        Genetic algorithm with elitism.
    - __create_individual(): Creates an individual for the genetic algorithm.
    - __length(individual): Calculates the length of an individual.
    - __weight_calculation(rout): Calculates the weight of a route.
    - __fitness(individuals): Calculates the fitness of a set of individuals.
    - __mutate_shuffle(individual, indpb): Applies mutation (shuffle) to an individual.
    - __ordered_crossover(individual1, individual2): Applies ordered crossover to two individuals.
    - __lokal_opt(routes, distances): Applies local optimization to a set of routes.
    - __find_duplicates(lst): Finds duplicates in a list.
    - __convert_length_to_duration(points, length, speed, string=True): Converts length to a time interval.
    - __get_solution(data, return_coordinates=False): Generates the coordinates of the routes based on the obtained solution.
    - _geocoding(data): Performs geocoding of addresses and prepares data for the knapsack problem.

    Attributes:
    - optimized_route (list): List containing the optimized route coordinates.
    - address_depo (pd.DataFrame): Depot information with default coordinates.
    - depo (pd.DataFrame): Current depot information used in the algorithm.
    - data (pd.DataFrame): Updated input data with geocoded information.
    - POPULATION_SIZE (int): Size of the population in the genetic algorithm.
    - MAX_GENERATIONS (int): Maximum number of generations in the genetic algorithm.
    - startV (int): Starting point for routes.
    - endV (int): Ending point for routes.
    - P_CROSSOVER (float): Crossover probability in the genetic algorithm.
    - P_MUTATION (float): Mutation probability in the genetic algorithm.
    - MAX_VOLUME (float): Maximum volume for each route.
    - MIN_VOLUME (float): Minimum volume for each route.
    - NUMBER_OF_ROUTES (int): Desired number of routes.
    - penalty (float): Penalty factor for weight deviation.
    - distance_matrix (np.ndarray): Distance matrix based on the input data.
    - LENGTH_D (int): Length of the distance matrix.
    - weights (list): List of weights based on the input data.
    - HALL_OF_FAME_SIZE (int): Size of the Hall of Fame in the genetic algorithm.
    - hof (tools.HallOfFame): Hall of Fame instance in DEAP toolbox.
    - toolbox (base.Toolbox): DEAP toolbox instance.
    - stats (tools.Statistics): DEAP statistics instance.

    Example Usage:
    ```python
    # Instantiate the solver
    solver = GeneticAlgorithmRoutingProblemSolver(data, VOLUME)

    # Solve the routing problem
    solver.genetic_route_solver()

    # Display information about the optimized route
    solver.genetic_route_information()
    ```
    """
    optimized_route = None
    address_depo = pd.DataFrame({"full address": ['Zawodzie 18, 02-981 Warszawa'],
                                "latitude": [52.183425], "longitude": [21.085442],
                                "Pojemnosc": 0 
                                   })
    
    def __init__(self, data, VOLUME, POPULATION_SIZE = 500, 
                 MAX_GENERATIONS = 1000, deviation_percentage_weight = 0.05, 
                 CROSSOVER = 0.75, P_MUTATION = 0.75, HALL_OF_FAME_SIZE = 5,
                 startV = 0, endV = "startV", penalty = 0.2, 
                depo = None, number_of_routes = None):

        DistanceMatrixCalculator.__init__(self, round_up=100_000)
        Geocoder.__init__(self, max_threads=10)

        self.depo = depo if depo else self.address_depo
        self.data = self.__check_data(data)
        
        self.POPULATION_SIZE = POPULATION_SIZE
        self.MAX_GENERATIONS = MAX_GENERATIONS

        self.startV = 0
        self.endV = self.startV if endV == "startV" else endV
        
        
        self.P_CROSSOVER = CROSSOVER
        self.P_MUTATION = P_MUTATION
        
        percentage_value = VOLUME * deviation_percentage_weight
        self.MAX_VOLUME = VOLUME + percentage_value
        self.MIN_VOLUME = VOLUME - percentage_value

        self.NUMBER_OF_ROUTES = number_of_routes if number_of_routes else ceil(self.data["Pojemnosc"].sum() / self.MAX_VOLUME )
        self.penalty = 0.2

        self.distance_matrix = self.create_manhattan_distance_matrix(self.data)
        self.LENGTH_D = len(self.distance_matrix)

        
        self.weights = self.data["Pojemnosc"].tolist()
        self.weights[self.startV] = 0
        
        self.HALL_OF_FAME_SIZE = HALL_OF_FAME_SIZE
        self.hof = tools.HallOfFame(self.HALL_OF_FAME_SIZE)

        self.toolbox = base.Toolbox()
        
        # Initialization DEAP
        # Checking if the class "FitnessMin" exists
        if not hasattr(creator, "FitnessMin"):
            creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        
        # Проверка, существует ли класс "Individual"
        if not hasattr(creator, "Individual"):
            creator.create("Individual", list, fitness=creator.FitnessMin)
        
        self.toolbox = base.Toolbox()
        self.toolbox.register("individualCreator", tools.initIterate, creator.Individual, self.__create_individual)
        self.toolbox.register("individual", self.toolbox.individualCreator)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        self.toolbox.register("evaluate", self.__fitness)
        self.toolbox.register("select", tools.selTournament, tournsize=HALL_OF_FAME_SIZE)
        self.toolbox.register("mate", self.__ordered_crossover)
        self.toolbox.register("mutate", self.__mutate_shuffle, indpb= self.P_MUTATION)
        self.stats = tools.Statistics(lambda ind: ind.fitness.values)
        self.stats.register("min", np.min)
        self.stats.register("avg", np.mean)

    def genetic_route_solver(self, verbose=True, return_coordinates = False):
        """
        Main function for solving the routing problem using a genetic algorithm.
    
        Parameters:
        - verbose (bool): Verbose output flag.
    
        Returns:
        list: List of DataFrames containing route coordinates.
        """
    
        # Initialize the initial population for the genetic algorithm
        population = self.toolbox.population(n=self.POPULATION_SIZE)
    
        # Run the genetic algorithm with elitism
        population, logbook = self.__eaSimpleElitism(population, self.toolbox,
                                              cxpb=self.P_CROSSOVER,
                                              mutpb=self.P_MUTATION,
                                              ngen=self.MAX_GENERATIONS,
                                              halloffame=self.hof,
                                              stats=self.stats,
                                              verbose=verbose
                                             )
    
        # Get the best individual from the Hall of Fame
        best = self.hof.items[0]
        
        # Apply local optimization to the best route
        self.optimized_route = self.__lokal_opt(best, self.distance_matrix)

        if verbose: self.genetic_route_information(verbose = verbose)
            
        # Return the coordinates of the optimized route
        return self.__get_solution(self.data, return_coordinates = return_coordinates)
 
    def genetic_route_information(self, verbose=True):
        """
        Calculate and display information about the optimized route obtained through the genetic algorithm.
    
        Parameters:
        - verbose (bool): Verbose output flag. If True, detailed information will be printed; otherwise, minimal information.
    
        Returns:
        tuple: A tuple containing the total distance, total load, and time taken for the optimized route.
        """
    
        if self.optimized_route:
            # Calculate the total distance of the best route
            total_distance = self.__fitness(self.optimized_route)[0] / 1000
    
            # Calculate the total weight gained during collection
            route_weights = [self.weights[point] for route in self.optimized_route for point in route]
    
            total_load = sum(route_weights) / 1000
    
            # Convert the length of the route to duration
            time = self.__convert_length_to_duration(points=sum(self.optimized_route, []), length=total_distance, speed=33, string=False)
    
            # Display verbose information
            if verbose:
                print(f"Found route №{len(self.optimized_route)} ")
                print(self.optimized_route)
                print()
                print(f"Approximate distance of all routes {total_distance} km ")
                print(f"Weight gained during collection is {total_load} m3")
                print(f"Approximate duration of route {time} h")
    
                for i, rout in enumerate(self.optimized_route):
                    print(f"weight rout {i}: {self.__weight_calculation(rout) / 1000} m3")
    
            return total_distance, total_load, time
        else:
            # Return zero values if no optimized route is available
            return 0, 0, (0, 0)
     
    def __check_data(self, data) -> pd.DataFrame:
        """
        Performs checks and updates the data .

        Returns:
        pd.DataFrame: Updated data.
        """
        if data["Pojemnosc"].isna().sum():
            if len(data["Pojemnosc"].unique()) == 2:
                data["Pojemnosc"].fillna(data["Pojemnosc"].mean(), inplace=True)
            else:
                nan_indices = data[data["Pojemnosc"].isna()].index
                raise ValueError(f"NaN detected in string {nan_indices[0]}")

        columns = data.columns
        if "longitude" not in columns and "latitude" not in columns :
            data = self._geocoding(data)


        # Concatenating the depot information with the geo_data DataFrame
        if data["full address"].ne(self.depo["full address"]).any():
            data = pd.concat([self.depo, data], axis=0, ignore_index=True).reset_index(drop = True)
            
        return data

    def __eaSimpleElitism(self, population, toolbox, cxpb, mutpb, ngen, stats=None,
                    halloffame=None, verbose= None, callback=None):
        
        """
        Genetic algorithm with elitism.
    
        Parameters:
        - population: Initial population.
        - toolbox: DEAP toolbox.
        - cxpb: Crossover probability.
        - mutpb: Mutation probability.
        - ngen: Number of generations.
        - stats: DEAP statistics.
        - halloffame: Hall of Fame.
        - verbose: Verbose output flag.
        - callback: Callback function.
    
        Returns:
        tuple: Final population and logbook.
        """
    
        # Initialization of the logbook object with a header
        logbook = tools.Logbook()
        logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])
        
        # Evaluation of individuals with invalid fitness
        invalid_ind = [ind for ind in population if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        # Update the Hall of Fame
        if halloffame is not None:
            halloffame.update(population)
        
        # Determine the size of the Hall of Fame
        hof_size = len(halloffame.items) if halloffame.items else 0
        
        # Record statistics for the first generation
        record = stats.compile(population) if stats else {}
        logbook.record(gen=0, nevals=len(invalid_ind), **record)
        
        # Display the logbook if verbose mode is enabled
        if verbose:
            print(logbook.stream)
        
        # Start the generational process
        for gen in range(1, ngen + 1):
            # Select the next generation of individuals
            offspring = toolbox.select(population, len(population) - hof_size)
        
            # Generate variation in the pool of individuals using crossover and mutation
            offspring = varAnd(offspring, toolbox, cxpb, mutpb)
        
            # Evaluate individuals with invalid fitness in the offspring
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
        
            # Add elite individuals to the offspring
            offspring.extend(halloffame.items)
        
            # Update the Hall of Fame with the offspring
            if halloffame is not None:
                halloffame.update(offspring)
        
            # Replace the current population with the offspring
            population[:] = offspring
        
            # Record statistics for the current generation
            record = stats.compile(population) if stats else {}
            logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        
            # Display the logbook if verbose mode is enabled
            if verbose:
                print(logbook.stream)
        
            # Call the user-defined callback function if provided
            if callback:
                callback[0](*callback[1])
        
        # Return the final population and logbook
        return population, logbook

    def __create_individual(self):
        """
        Creates an individual for the genetic algorithm.
    
        Returns:
        creator.Individual: Created individual.
        """
        # Initialize an empty list to store the individual's routes
        current_route = []
        # Create a list of indices representing all points in the distance matrix
        distance_len = [i for i in range(self.LENGTH_D)]
    
        # Iterate over the desired number of routes
        for i in range(self.NUMBER_OF_ROUTES):
            # Initialize an empty list to store a single route
            route = []
            # Iterate to select points for the route
            for _ in range((self.LENGTH_D // self.NUMBER_OF_ROUTES + 1)):
                # Check if there are points available
                if not distance_len:
                    break
    
                # Randomly select a point from the available points
                point = np.random.choice(distance_len)
    
                # Exclude the starting point
                if point != self.startV:
                    route.append(point)
                    distance_len.remove(point)
    
            # Check if it's not the last route
            if i < self.NUMBER_OF_ROUTES - 1:
                # Append the current route with the starting and ending points
                current_route.append([self.startV] + route[:-2] + [self.endV])
            else:
                # For the last route, shuffle remaining values and append with starting and ending points
                remaining_values = [i for i in range(self.LENGTH_D) if i not in sum(current_route, []) and i != self.startV]
                np.random.shuffle(remaining_values)
                current_route.append([self.startV] + remaining_values + [self.endV])
    
        # Return the created individual as a DEAP creator.Individual instance
        return creator.Individual(current_route)

    def __length(self, individual):
        """
        Calculates the length of an individual.
    
        Parameters:
        - individual (creator.Individual): Individual.
    
        Returns:
        float: Length of the individual.
        """
        distance = 0
        
        for i in range(len(individual) - 1):
            distance += self.distance_matrix[individual[i]][individual[i + 1]]
        
        return distance

    def __weight_calculation(self, rout):
        """
        Calculates the weight of a route.
    
        Parameters:
        - rout (list): Route.
    
        Returns:
        float: Weight of the route.
        """
        if None in rout:
            print("rout:", rout)
            
        return sum([self.weights[i] for i in rout])

    def __fitness(self, individuals):
        """
        Calculates the fitness of a set of individuals.
    
        Parameters:
        - individuals: Set of individuals.
    
        Returns:
        tuple: Fitness values.
        """
        # Initialize an empty list to store distances for all individuals
        all_distances = []
    
        # Iterate over each individual in the set
        for individual in individuals:
            # Initialize variables for distance and weight
            distance = 0
            weight = self.__weight_calculation(individual)
    
            # Calculate the total distance for the individual
            distance = self.__length(individual)
    
            # Penalize if the weight is below the minimum volume
            if weight < self.MIN_VOLUME:
                distance += self.MAX_VOLUME / self.penalty
            # Penalize if the weight exceeds the maximum volume
            elif weight > self.MAX_VOLUME:
                distance += self.MAX_VOLUME * self.penalty
    
            # Append the total distance to the list
            all_distances.append(distance)
    
        # Return the sum of all distances as a single-value tuple
        return sum(all_distances),

    def __mutate_shuffle(self, individual, indpb):    
        """
        Applies mutation (shuffle) to an individual.
    
        Parameters:
        - individual (creator.Individual): Individual.
        - indpb: Probability of mutation.
    
        Returns:
        creator.Individual: Mutated individual.
        """
        mutated_gene = []
    
        for mutated_individual in individual.copy():
            
            # Internal mutation
            for i in range(len(mutated_individual)):
               # Generate a random number from 0 to 1 and check if the mutation is running for the current gene
                if random.random() < indpb:
                    # Select a random index for exchange
                    if len(mutated_individual) > 4:
                        swap_index = random.randint(1, len(mutated_individual) - 2)
                        # Exchange values between the current gene and the gene at the selected index
                        if i != 0 and i != len(mutated_individual) - 1:
                            mutated_individual[i], mutated_individual[swap_index] = mutated_individual[swap_index], mutated_individual[i]
            
            mutated_gene.append(mutated_individual)
            
        # external mutation
        length_smallest_gene =  min([ len(gene) for gene in mutated_gene ]) 
        
        for index in range(len(mutated_gene) - 1):
            if random.random() < indpb:
                if length_smallest_gene > 4:
                    swap_index = random.randint(1, length_smallest_gene - 2)
                    if swap_index != 0 and swap_index != len(mutated_gene[index]):
                        
                     mutated_gene[index][swap_index], mutated_gene[index + 1][swap_index] = mutated_gene[index + 1][swap_index], mutated_gene[index][swap_index] 
            
        return creator.Individual(mutated_gene),

    def __ordered_crossover(self, individual1, individual2):
        """
        Applies ordered crossover to two individuals.
    
        Parameters:
        - individual1 (creator.Individual): First individual.
        - individual2 (creator.Individual): Second individual.
    
        Returns:
        tuple: Two descendants.
        """
        # Initialize empty lists for descendants
        descendant1 = []
        descendant2 = []
    
        # Iterate over pairs of genes from the parents
        for parent in zip(individual1, individual2):
            parent1 = parent[0]
            parent2 = parent[1]
    
            try:
                # Determine the size of the parents and a crossover point
                size = min(len(parent1), len(parent2))
                index = np.random.randint(1, size - 1)
    
                # Check for duplicate elements in parents
                if len(set(parent1[1:-1] + parent2[1:-1])) == len(parent1) + len(parent2) - 3:
                    raise ValueError("Duplicate elements detected in parent")
    
                # Initialize empty lists for children
                child1 = [None] * len(parent1)
                child2 = [None] * len(parent2)
    
                # Set the starting and ending points for children
                child1[0] = 0
                child2[0] = 0
                child1[-1] = 0
                child2[-1] = 0
    
                # Apply ordered crossover
                if parent1[index] in parent2 and parent2[index] in parent1:
                    child1[index] = parent2[index]
                    child2[index] = parent1[index]
    
                # Complete the children by filling in the remaining values
                for i, value in enumerate(parent1):
                    if value != 0 and None in child1:
                        nan_index = child1.index(None)
                        child1[nan_index] = value if value not in child1 else None
    
                for i, value in enumerate(parent2):
                    if value != 0 and None in child2:
                        nan_index = child2.index(None)
                        child2[nan_index] = value if value not in child2 else None
    
                # Append the children to the descendants
                descendant1.append(child1)
                descendant2.append(child2)
    
                # Check for None values in descendants
                if any(None in rout for rout in descendant1):
                    print("Descendant 1 contains None values.")
                    raise ValueError
                if any(None in rout for rout in descendant2):
                    print("Descendant 2 contains None values.")
                    raise ValueError
    
            except ValueError as e:
                # Handle ValueError by reusing parents
                print("ValueError. Reusing parents.")
                descendant1.append(parent1)
                descendant2.append(parent2)
    
        # Return descendants as DEAP creator.Individual instances
        return creator.Individual(descendant1), creator.Individual(descendant2)

    def __lokal_opt(self, routes, distances):
        """
        Applies local optimization to a set of routes.
    
        Parameters:
        - routes (list): List of routes.
        - distances: Distance matrix.
    
        Returns:
        list: Optimized routes.
        """
        # Initialize an empty list to store optimized routes
        result = []
        
        # Iterate over each route in the given set of routes
        for route in routes:
            # Flag to track improvement in the route
            improved = True
            
            # Continue optimizing the route until no further improvement is possible
            while improved:
                improved = False
                
                # Iterate over pairs of indices in the route
                for i in range(1, len(route) - 2):
                    for j in range(i + 1, len(route)):
                        # Skip pairs with a difference of 1 (2-opt changes for consecutive cities are equivalent to a simple swap)
                        if j - i == 1:
                            continue
                        
                        # Apply 2-opt exchange to create a new route
                        new_route = route[:i] + route[i:j][::-1] + route[j:]
                        new_distance = self.__weight_calculation(new_route)
                        
                        # Check if the new route is better than the current route
                        if new_distance < self.__weight_calculation(route):
                            route = new_route
                            improved = True
                            
            # Append the optimized route to the result list
            result.append(route)
        
        # Return the list of optimized routes
        return result

    def __find_duplicates(self, lst):
        """
        Finds duplicates in a list.
    
        Parameters:
        - lst: Input list.
    
        Returns:
        dict: Dictionary of duplicate counts.
        """
        seen = set()
        duplicates = {}
        
        for value in lst:
            if value != 0:
                if value in seen:
                    if value in duplicates:
                        duplicates[value] += 1
                    else:
                        duplicates[value] = 2  
                else:
                    seen.add(value)
    
        return duplicates

    def __convert_length_to_duration(self, points, length, speed, string=True):
        """
        Converts length to a time interval.
    
        Args:
            length (float): Length to convert.
            speed (float): Speed for the conversion.
            string (bool, optional): If True, returns a string in the format 'hours:minutes'.
                If False, returns a tuple of hours and minutes. Defaults to True.
    
        Returns:
            str or tuple: Converted duration in hours:minutes format if string is True,
                          otherwise, a tuple of hours and minutes.
        """
        # Convert length to time interval
        duration_hours = length / speed
        duration_hours = duration_hours + ((len(points) * 2.2 ) / 60 )
        
        # Separate hours and minutes
        hours = int(duration_hours)
        minutes = int((duration_hours - hours) * 60)
        
        if string : return f'{hours}:{minutes}'
        else: return hours, minutes

    def __get_solution(self, data, return_coordinates = False): 
        """
        Generates the coordinates of the routes based on the obtained solution.
    
        Parameters:
        data (pd.DataFrame): The DataFrame containing latitude and longitude data.
        route (list): The list of indices representing the route.
    
        Returns:
        pd.DataFrame: A DataFrame containing the route coordinates.
        """
        
        all_routs = []
        count = 1 
        
        if not return_coordinates:
            for route in self.optimized_route:
                # Extract latitude and longitude coordinates for each index in the route
                df = pd.DataFrame([data.iloc[i] for i in route])
                df = df.reset_index(drop = True)
                df = df.assign(route=count)  
            
                all_routs.append(df)
                count += 1
    
            data = pd.concat(all_routs).reset_index(names= "index").sort_values(["route", "index"])
            data = data.drop("index", axis = 1)
        
            return data
        else :
            for route in routes:
                # Extract latitude and longitude coordinates for each index in the route
                df = pd.DataFrame()
                
                df["latitude"] = [data.loc[route[i], "latitude"] for i in range(len(route))]
                df["longitude"] = [data.loc[route[i], "longitude"] for i in range(len(route))]
                df = df.assign(route=count)  
                all_routs.append(df)
                count += 1
                
            return all_routs

    def _geocoding(self, data):
        """
        Performs geocoding of addresses and prepares data for the knapsack problem.

        Returns:
            DataFrame: Processed data with geocoded information.
        """
        print("geocoding process")

        data_for_geocoding = self.address_collector(data.copy())
        data_for_geocoding.reset_index(drop = True, inplace = True)
        
        unique_values = data_for_geocoding["full address"].drop_duplicates()
        unique_indices = unique_values.index.to_numpy()

        unique_data_for_geocoding = data_for_geocoding.iloc[unique_indices]
                
        self.geocode_addresses(unique_data_for_geocoding)
        self.save_results_to_csv()

        # Get geography data and merge it with the source data
        result_coding = self.get_results()

        result_coding = pd.DataFrame(result_coding)
        
        merge_data = data_for_geocoding.merge(result_coding, how="inner", left_on="full address", right_on="address").drop("address", axis = 1)
        

        # Prepare data for the task
        columns_to_replace = ["Pojemnosc", "longitude", "latitude"]
        merge_data[columns_to_replace] = merge_data[columns_to_replace].replace('', np.nan)

        merge_data = merge_data.dropna(subset=["longitude", "latitude"])
        merge_data = merge_data.reset_index(drop=True)

        # Display information about the number of found and missing data
        missing = len(data_for_geocoding) - len(merge_data)
        if missing:
            print(f"{missing} missing data found in the data.")
            # Display the addresses that were not successfully geocoded, sorted by "full address"
            missing_addresses = data_for_geocoding.loc[~data_for_geocoding["full address"].isin(merge_data["full address"]), "full address"]
            missing_addresses_sorted = missing_addresses.sort_values()
            print("Addresses missing data:")
            display(missing_addresses_sorted)
        else:
            print("All data successfully geocoded.")
        
        return merge_data




# Generate routes 
def generate_routes (data, solver_type = "or_tools", capacity = 21_000, 
                     revert= False, save = True, col_name = "groups",
                     map = True, batch_size = 50):
    """
    This function generates routes for waste collection based on the provided data using either the Genetic Algorithm solver or the Google OR-Tools solver.
    
    Parameters:
    - data (pd.DataFrame): DataFrame containing route information.
    - solver_type (str): Type of solver to use - "genetic" or "or_tools". Default is "or_tools".
    - capacity (int): Vehicle capacity for waste collection in liters. Default is 21,000 liters.
    - revert (str or bool): If "stat", return only the route statistics; if "data", return the concatenated route dataframes; 
                           if "all_info", return both route dataframes and statistics. Default is False.
    - save (bool): Specifies whether to save the generated routes and statistics. Default is True.
    - col_name (str): Name of the column containing route groups. Default is "groups".
    - map (bool): Specifies whether to generate maps for the routes. Default is True.
    - batch_size (int): Number of points to include in each batch for routing. Default is 50.
    
    Returns:
    - If revert is "stat": DataFrame with route statistics.
    - If revert is "data": Concatenated DataFrame with all route coordinates.
    - If revert is "all_info": Tuple containing concatenated DataFrame with all route coordinates and DataFrame with route statistics.
    - If revert is False: None.
    
    """
    today_date = datetime.now().date()
    if map: 
        rim = RouteInfoAndMap()
        maps = []
        
    routs_statistics = {}
    list_data = []

    folder_path = f"data/routes/{solver_type}_{col_name}_{today_date}"
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    group_name = data[col_name].unique()
    
    for name in tqdm(group_name):
        df = data.loc[data[col_name] == name ]

        if df.empty:
            print()
            print(f"The dataset named '{name}' is empty, check if you have entered the group names for division correctly.")
            continue
        else:
            if solver_type == "genetic":
                try:
                    # Using Genetic Algorithm solver
                    gen_solver = GeneticAlgorithmRoutingProblemSolver(df, capacity, MAX_GENERATIONS=10)
                    route_coordinates = gen_solver.genetic_route_solver(verbose=False, return_coordinates=False)
                    distance, volume, (hours, minutes) = gen_solver.genetic_route_information(verbose=False)
                except ValueError :
                    print()
                    print(f"Note that data frame {name} is very small for solver {solver_type} solver or contains a Nan value and will be skipped, check the data or try a different solver to include this data.")
                    continue
                    
            elif solver_type == "or_tools":
                try:
                    # Using Google OR-Tools solver
                    ort_solver = OrToolsRoutingProblemSolver(df, time_limit_seconds=60, vehicle_capacities=capacity)
                    route_coordinates = ort_solver.ort_route_solver()
                    distance, volume, (hours, minutes) = ort_solver.ort_route_information(verbose=False)
                except ValueError :
                    print()
                    print(f"Note that the {name} data frame likely contains a Nan value and will be passed by the {solver_type} solver. Make sure the data is correct so the solver can process it.")
                    print()
                    
                    continue
            else:
                raise TypeError("Wrong router type. Please select one of the available routers: genetic|or_tools")

        list_data.append(route_coordinates)

        
        if save:
            # Saving route coordinates to CSV
            route_coordinates.to_csv(f"{folder_path}/rout_{name}.csv", index=False)

        if not map:
            # Collecting route statistics
            routs_statistics[name] = {"distance" : distance, 
                                      "volume" : volume, 
                                      "hours" : hours, 
                                      "minutes" : minutes}
        if map:
            try:
                maps_folder_path = folder_path + "/map"
                if not os.path.exists(maps_folder_path):
                    os.makedirs(maps_folder_path)
            
                map, distance, time = rim.generate_routes_info_and_maps(route_coordinates,
                                                                        revert= "stats",
                                                                        batch_size=batch_size,
                                                                        save=True,
                                                                        save_path= maps_folder_path,
                                                                        name_map= f'{name}_map' )
                maps.append(map)
                            
                routs_statistics[name] = {"distance" : distance, 
                                          "volume" : volume, 
                                          "time" : time }
            except ApiError as e:
                print()
                print(f"{e} \nthis error says that the parameter has an incorrect value or format, \ntry changing batch_size if this does not help then refer to the Open Route Service documentation.")
                print("https://openrouteservice.org/")
                return None
    
    # Creating DataFrame with route statistics
    routs_statistics = pd.DataFrame(routs_statistics).T
    
    if save:
        # Saving route statistics to CSV        
        folder_path = f"data/routes/routs_statistics"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        routs_statistics.to_csv(f"{folder_path}/{col_name}_{solver_type}_routs_statistics_{today_date}.csv", index_label="group_name")

    if revert == "stat":
        return routs_statistics
    elif revert == "data":
        return pd.concat(list_data).reset_index(drop = True)
    elif  revert == "all_info":
        return pd.concat(list_data).reset_index(drop = True), routs_statistics
    else:
        None


