from lib.imports import *

class VisualizerDataFilling:
    """
    The VisualizerDataFKnapsackSolver class provides methods for visualizing data related to the F-Knapsack Solver.

    Attributes:
    - order_dict (dict): A dictionary specifying the order of elements for plotting.

    Methods:
    - plot_faction(data, x="groups", save=False, return_barplot=False):
        Plot a bar chart showing the count of elements in each faction group.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.
        - x (str): Column to plot on the x-axis.
        - save (bool): Whether to save the plot as an image (default is False).
        - return_barplot (bool): Whether to return the barplot object (default is False).

    - plot_week(data, x="groups", save=False):
        Plot a bar chart showing the count of elements in each week for each waste code.

        Parameters:
        - data (pd.DataFrame): DataFrame containing the data.
        - x (str): Column to plot on the x-axis.
        - save (bool): Whether to save the plot as an image (default is False).
    """
    def __init__(self, order_dict={'Pn': 1, 'Wt': 2, 'Sr': 3, 'Cz': 4, 'Pt': 5, 'So': 6}):
        """
        Initialize the VisualizerDataFKnapsackSolver class.

        Parameters:
        - order_dict (dict): A dictionary specifying the order of elements for plotting.
        """
        self.order_dict = order_dict

    def plot_faction(self, data, x="groups", save=False, 
                    return_barplot = False):
        """
        Plot a bar chart showing the count of elements in each faction group.
    
        :param data: DataFrame containing the data.
        :param x: Column to plot on the x-axis.
        :param save: Whether to save the plot as an image (default is False).
        """
        
        factions = sorted( i for i in data["groups"].str[6:].unique() if isinstance(i, str))
        for faction in factions:
            filtered_rows = data[data["groups"].str[6:] == faction].reset_index(drop=True)
            filtered_rows["order_str"] = filtered_rows["groups"].str[1:3].map(self.order_dict)
            filtered_rows["order_int"] = filtered_rows["groups"].str[0].astype("int")
            filtered_rows = filtered_rows.sort_values(by=["order_int", "order_str"]).reset_index(drop=True)

            my_palette = sns.color_palette("viridis", n_colors=len(filtered_rows["order_str"].unique())) * len(
                filtered_rows["order_int"].unique())
            plt.figure(figsize=(15, 10))
            
            sns.barplot(
                        x=x,
                        y=filtered_rows.index,
                        hue=x,
                        data=filtered_rows,
                        palette=my_palette,
                        errorbar=None,
                        estimator="count",
                        width=0.8,
                    )

            plt.title(f'Number of elements in faction group {faction}')
            plt.xlabel(x)
            plt.ylabel("Number of elements")
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()
            
            filtered_rows = filtered_rows.drop(columns=["order_str", "order_int"])

            if save:
                plt.savefig(f'pictures/plot_faction{faction}.png')

            plt.show()

    def plot_week(self, data, x="groups", save=False):
        """
        Plot a bar chart showing the count of elements in each week for each waste code.
    
        :param data: DataFrame containing the data.
        :param x: Column to plot on the x-axis.
        :param save: Whether to save the plot as an image (default is False).
        """
        frequencies = sorted( i for i in data["Częstotliwość"].unique() if isinstance(i, str))
        
        weeks_dict = {}

        for day in frequencies:
            week_number = int(day[0])
            if week_number not in weeks_dict:
                weeks_dict[week_number] = []
            weeks_dict[week_number].append(day)

        for frequency_chunk in weeks_dict.values():
            filtered_rows = data[data[x].astype(str).str[:3].isin(frequency_chunk)].reset_index(drop=True)
            filtered_rows["order"] = filtered_rows[x].str[1:3].map(self.order_dict)
            filtered_rows = filtered_rows.sort_values(by=["Kod odpadu", "order"]).drop(columns="order").reset_index(drop=True)

            plt.figure(figsize=(15, 10))
            sns.barplot(
                x=x,
                y=filtered_rows.index,
                hue=x,
                data=filtered_rows,
                palette="viridis",
                errorbar=None,
                estimator="count",
                width=0.8,
            )

            plt.title(f'Number of elements in different groups in week {frequency_chunk[0][0]}')
            plt.xlabel(x)
            plt.ylabel("Number of elements")
            plt.xticks(rotation=45, ha="right")

            plt.tight_layout()

            if save:
                plt.savefig(f'pictures/plot_week {filtered_rows[0][0]}.png')

            plt.show()

class RouteInfoAndMap:
    """
    The RouteInfoAndMap class provides functionality to generate route information and maps based on input data.

    Attributes:
    - open_route_token_path (str): Path to the JSON file containing the token for the OpenRouteService.
      Default is "data/token_OpenRouteService.json".

    Methods:
    - generate_route_info_and_map(df, batch_size=50, save=False, save_path="data", name_map="map",
                                   open_route_token_path=None):
        Generates route information and a map based on input data and optionally saves the map as an HTML file.

        Parameters:
        - df (pd.DataFrame): Input data containing columns "latitude" and "longitude" representing geographic coordinates.
        - open_route_token_path (str): Path to the JSON file containing the token for the OpenRouteService.
          Default is None (uses the class attribute if not provided).
        - save (bool): Specifies whether to save the map as an HTML file.
        - save_path (str): Path to save the HTML file if save=True.
        - name_map (str): Name of the HTML file if save=True.
        - batch_size (int): Number of points to include in each batch for routing.

        Returns:
        - folium.folium.Map: Folium map with the added route.
        - float: Total length of the route in kilometers.
        - float: Total duration of the route in hours.

    - generate_routes_info_and_maps(data, delay=2.2, verbose=False, col_name="groups", revert=False,
                                     file_path="data/token_OpenRouteService.json", save=False, save_path="data",
                                     name_map="map", batch_size=50):
        Calculate and print statistics for each route in the provided data.

        Parameters:
        - data (pd.DataFrame): DataFrame containing route information.
        - delay (float): Additional delay time in minutes for each stop. Default is 2.2 minutes.
        - verbose (bool): If True, print detailed information for each route. Default is False.
        - col_name (str): Name of the column to use for grouping routes.
        - revert (str or bool): If "maps", return all generated maps; if "stats", return route statistics DataFrame;
                                if "all_info", return both maps and statistics. Default is False.
        - file_path (str): Path to the JSON file containing the token for the OpenRouteService.
          Default is "data/token_OpenRouteService.json".
        - save (bool): Specifies whether to save the maps as HTML files.
        - save_path (str): Path to save the HTML files if save=True.
        - name_map (str): Base name of the HTML files if save=True.
        - batch_size (int): Number of points to include in each batch for routing.

        Returns:
        - If revert is "maps": List of generated Folium maps.
        - If revert is "stats": DataFrame with route statistics.
        - If revert is "all_info": Tuple containing DataFrame with route statistics and list of maps.
        - If revert is False: None.
    """
    def __init__(self, open_route_token_path = "data/token_OpenRouteService.json"):
        """
        Initialize the RouteInfoAndMap class.

        Parameters:
        - open_route_token_path (str): Path to the JSON file containing the token for the OpenRouteService.
        Default is "data/token_OpenRouteService.json".
        """
        self.open_route_token_path = open_route_token_path

    def generate_interactive_route_map(self, df, save=False, delay=2.2, save_path="data", name_map="map"):
        """
        Generate an interactive route map with dynamic updates.

        Parameters:
        - df (pd.DataFrame): DataFrame containing location data.
        - save (bool): Whether to save the generated map.
        - delay (floating point): delay time for loading the transport.
        - save_path (str): Directory path for saving the map.
        - name_map (str): Name for the saved map.

        Returns:
        - m (folium.Map): The generated Folium map.
        - total_length (float): Total distance of the generated routes.
        - total_duration (float): Total duration of the generated routes.
        """
        total_length = 0.0
        total_duration = 0.0
        volume = 0
        all_row = len(df)
        s = 0
        e = 0
        batch_routes = None
        
        # Initialize the Folium map
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        
        # Use default token path if not provided
        if not self.open_route_token_path:
            self.open_route_token_path = "path/to/default/token.json"
    
        # Read OpenRouteService token from file
        with open(self.open_route_token_path, 'r') as f:
            data = json.load(f)
            token = data["token"]
    
        # Create OpenRouteService client
        client = openrouteservice.Client(key=token)

        print("To add a point, click on '+' or 'esc' from to exit")
        while True:
            if keyboard.is_pressed('+'):
                
                # Clear the output for a cleaner display
                clear_output(wait=True) 
                e = s + 2
                if e <= all_row:
                    
                    # Extract coordinates for the current batch
                    batch_df = df.iloc[s: e]
                    batch_coordinates = [[lon, lat] for lat, lon in zip(batch_df['latitude'], batch_df['longitude'])]
                    
                    # Get directions for the current batch
                    batch_routes = client.directions(coordinates=batch_coordinates, profile='driving-car', format='geojson')
        
                    # Add polyline for the current batch
                    polyline = folium.PolyLine(locations=[list(reversed(coord)) for coord in batch_routes['features'][0]['geometry']['coordinates']],
                                               color="red", weight=2.5, opacity=10)
                    polyline.add_to(m)
        
                    # Add markers for each point in the current batch
                    for i in batch_coordinates:
                        folium.Marker(location=[i[1], i[0]], popup=f"Point {i}").add_to(m)

                    # Display the updated map
                    display(m)
                    s += 1
                
                # Update total distance, duration, and volume
                if "features" in batch_routes and batch_routes["features"]:
                    last_feature = batch_routes["features"][-1]
                    if "properties" in last_feature and "summary" in last_feature["properties"]:
                        total_length += last_feature["properties"]["summary"].get("distance", 0)
                        total_duration += last_feature["properties"]["summary"].get("duration", 0)
                
                if e <= all_row:
                    volume += df.iloc[e]["Pojemnosc"]
                
                # Print current statistics
                print("length:", total_length)
                print("duration:", (total_duration / 3600) + ((e * delay) / 60))
                print("volume:", volume)
                
                time.sleep(0.2)
        
            elif keyboard.is_pressed('Esc'):
                # Clear the output and generate the final route info and map
                clear_output(wait=True)
                self.generate_route_info_and_map(df = df, 
                                                 batch_size=10, 
                                                 save=save, delay = delay, 
                                                 save_path=save_path, 
                                                 name_map=name_map)
                break
    
        # Save the map if required
        if save:
            m.save(f'{save_path}/{name_map}.html')
    
        # Convert total length and duration to appropriate units
        total_length = total_length / 1000
        total_duration = total_duration / 3600
        total_duration = total_duration + ((len(df) * delay) / 60)
    
        return m, total_length, total_duration

    def generate_route_info_and_map(self, df, batch_size=50, save=False,
                                    delay = 2.2, save_path="data", name_map="map",
                                    open_route_token_path = None):
    
        """
        Generates route information and a map based on input data and optionally saves the map as an HTML file.
        
        Args:
        df (pd.DataFrame): Input data containing columns "latitude" and "longitude" representing geographic coordinates.
        open_route_token_path (str): Path to the JSON file containing the token for the OpenRouteService.
        save (bool): Specifies whether to save the map as an HTML file.
        save_path (str): Path to save the HTML file if save=True.
        name_map (str): Name of the HTML file if save=True.
        batch_size (int): Number of points to include in each batch for routing.
        delay (floating point): delay time for loading the transport.
    
        Returns:
        folium.folium.Map: Folium map with the added route.
        float: Total length of the route in kilometers.
        float: Total duration of the route in hours.
    
        Example:
        >>> data = pd.DataFrame({'latitude': [52.2297, 52.4069, 52.5200, 52.5200, 52.3702], 'longitude': [21.0122, 21.9189, 13.4050, 13.4050, 4.8952]})
        >>> create_map_with_route(data, save=True, save_path="data", name_map="example_map")
        """
    
        m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=11)
        total_length = 0.0
        total_duration = 0.0
    
        if not open_route_token_path:
            open_route_token_path = self.open_route_token_path
            
        with open(open_route_token_path, 'r') as f:
            data = json.load(f)
            token = data["token"]
    
        client = openrouteservice.Client(key=token)
    
        # Split the coordinates into batches
        for i in range(0, len(df), batch_size):
            batch_df = df.iloc[i:i+batch_size]
            batch_coordinates = [[lon, lat] for lat, lon in zip(batch_df['latitude'], batch_df['longitude'])]
    
            batch_routes = client.directions(coordinates=batch_coordinates, profile='driving-car', format='geojson')
    
            # Add polyline for each batch
            polyline = folium.PolyLine(locations=[list(reversed(coord)) for coord in batch_routes['features'][0]['geometry']['coordinates']],
                                       color="red", weight=2.5, opacity=10)
            polyline.add_to(m)


            # Update total distance, duration, and volume
            if "features" in batch_routes and batch_routes["features"]:
                last_feature = batch_routes["features"][-1]
                if "properties" in last_feature and "summary" in last_feature["properties"]:
                    total_length += last_feature["properties"]["summary"].get("distance", 0)
                    total_duration += last_feature["properties"]["summary"].get("duration", 0)
                        
    
        # Adding markers to the map
        for i, row in df.iterrows():
            folium.Marker(location=[row['latitude'], row['longitude']], popup=f"Point {i}").add_to(m)
    
        if save:
            m.save(f'{save_path}/{name_map}.html')

        total_length = total_length/ 1000
        # Adjust duration with additional delay time for each stop
        total_duration = total_duration / 3600
        total_duration = total_duration + ((len(df) * delay) / 60)
        
        return m, total_length, total_duration 

    def generate_routes_info_and_maps (self, data, delay = 2.2, verbose = False, 
                                       col_name = "groups", revert= False,
                                      file_path="data/token_OpenRouteService.json", 
                                      save=False, save_path="data", name_map="map", 
                                      batch_size=50):
        """
        Calculate and print statistics for each route in the provided data.
    
        Parameters:
        - data (pd.DataFrame): DataFrame containing route information.
        - delay (float): Additional delay time in minutes for each stop. Default is 2.2 minutes.
        - verbose (bool): If True, print detailed information for each route. Default is False.
        - revert (str or bool): If "maps", return all generated maps; if "stats", return route statistics DataFrame;
                                if "all_info", return both maps and statistics. Default is False.
    
        Returns:
        - If revert is "maps": List of generated Folium maps.
        - If revert is "stats": DataFrame with route statistics.
        - If revert is "all_info": Tuple containing DataFrame with route statistics and list of maps.
        - If revert is False: None.
        
        """
        
        # Initialize empty dictionary to store route statistics
        routs_statistics = {}
        
        # Initialize total length and duration variables
        total_length = 0
        total_duration = 0
        
        # Initialize empty list to store all generated maps
        all_maps = []
        
        # Get unique routes from the provided data
        routs = data["route"].unique()
        
        # Iterate through each route
        for i, route in enumerate(routs):
            # Extract data for the current route
            route_data = data[data["route"] == route]
            
            # Generate map and calculate route length and duration
            maps, length, duration = self.generate_route_info_and_map(route_data, save=save, 
                                                                 delay = delay,
                                                                 save_path=save_path, 
                                                                 name_map=name_map + f"_rout_{i}",
                                                                 batch_size=batch_size )
            
            # Print route information if verbose is True
            if verbose:
                print(f'Approximate distance of route {i + 1}: {length} kilometers')
                print(f'Approximate duration of the route {i + 1}: {duration} hours')
                print()
    
            # Collect route statistics
            routs_statistics[route_data[col_name].unique()[0]] = {"distance": length, 
                                                                  "volume": route_data["Pojemnosc"].sum(), 
                                                                  "duration": duration}
            
            # Accumulate total length and duration
            total_length += length
            total_duration += duration
            
            # Append the generated map to the list
            all_maps.append(maps)
            
        # Print total route information if verbose is True
        if verbose:
            print("Total:\n", 
                  "Approximate distance of all routes on the map:", total_length,
                  "\nApproximate duration of the all routes:", total_duration)
    
        # Return based on the specified revert option
        if revert == "maps":
            return all_maps
        elif revert == "stats":
            return all_maps, length, duration
        elif  revert == "all_info":
            return pd.DataFrame(routs_statistics), all_maps
        else:
            # Return None if revert is not specified
            None



def custom_style_pivot_table(pivot_table, sorted_by="D", save=False, 
                             save_name="sum_waste_capacity_groups"):
    """
    Apply custom styling to a pivot table.

    Parameters:
    - pivot_table (pd.DataFrame): The input pivot table to be styled.
    - sorted_by (str, optional): The sorting criterion. 
      "D" for day sorting, "W" for week sorting, and default for the default sorting logic.
    - save (bool, optional): Whether to save the styled DataFrame as an HTML file.
    - save_name (str, optional): The name of the HTML file if save is True.

    Returns:
    - pd.io.formats.style.Styler: Styled DataFrame.

    Example:
    >>> custom_style_pivot_table(my_pivot_table, sorted_by="D", save=True, save_name="styled_pivot")
    """

    # Mapping of days to numerical order
    order_dict = {'Pn': 1, 'Wt': 2, 'Sr': 3, 'Cz': 4, 'Pt': 5, 'So': 6, "ll": 7}

    if sorted_by.lower() == "d":
        # Sorting logic for day
        column_order = sorted(pivot_table.columns, key=lambda x: (order_dict[x[1:]]))
    elif sorted_by.lower() == "w":
        # Sorting logic for week
        column_order = sorted(pivot_table.columns, key=lambda x: (x[0]))
    else:
        # Default sorting logic
        column_order = sorted(pivot_table.columns, key=lambda x: (x[0], order_dict[x[1:]]))

    # Sort the columns in the pivot table
    pivot_table_sorted = pivot_table[column_order]

    # Apply background gradient to the sorted pivot table
    styled_table = pivot_table_sorted.style.background_gradient(cmap="YlGnBu", axis=None, vmin=0, vmax=pivot_table_sorted.mean().mean())

    # Save the styled DataFrame as an HTML file with rendered links
    if save:
        styled_table.to_html(f"pictures/{save_name}.html", render_links=True)

    return styled_table

def colored_marker_map(data, marker_column=None):
    """
    Creates a map with colored markers based on data from a DataFrame.

    Args:
        data (pandas.DataFrame): DataFrame containing geographical coordinate data.
        marker_column (str): Column with data to be used as labels for the markers.

    Returns:
        folium.Map: Completed map with colored markers.

    This function generates an interactive map with colored markers based on the data in the DataFrame.
    Each marker represents a single point on the map. You can specify the column to be used as
    labels for the markers, allowing additional information to be displayed when hovering over a marker.

    Note:
    - Make sure you have the 'folium' library installed before using this function.
    - The function generates random colors for markers and assigns colors based on the values in the 'marker_column'.
    - The map is generated based on the first point in the DataFrame as the initial location.
    """
    # Function to generate a random color
    def random_color():
        return "#{:02x}{:02x}{:02x}".format(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
    # Create a map with the initial location based on the first point in the DataFrame
    mapa = folium.Map(location=[data["latitude"].iloc[0], data["longitude"].iloc[0]], zoom_start=10)
    
    # Check if the marker_column is specified
    if marker_column:
        unique_labels = data[marker_column].unique()
    else:
        return None
    
    # Create a color palette
    label_colors = {label: random_color() for label in unique_labels}
    
    # Add colored markers to the map
    for index, row in data.iterrows():
        label = row[marker_column]
        color = label_colors.get(label, 'gray')  # Use a random color if the label is new
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=5,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=label
        ).add_to(mapa)
    
    return mapa





