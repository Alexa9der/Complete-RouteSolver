# Truck Route Optimization Project

## Project Overview

This project aims to improve the efficiency of cargo collection by optimizing routes and collection frequency using Data Science methods and techniques.

## Collection and Preparation of Data

Data should be collected on current collection routes, collection frequency, delivery times, and other relevant factors.

## Simulation and Optimization

Application of route optimization algorithms such as routing algorithms and genetic algorithms to minimize the time and resources spent on cargo collection.

Analyze collection frequency data to determine the optimal schedule based on cargo volumes and peak periods.

## Visualization and Results

Creation of graphs and visualizations to visually represent optimized routes and frequency of cargo collection.

## Iterative Process

After implementing the proposed optimizations, continue to monitor the data and make adjustments to strategies if necessary.

# Project Structure

The project is organized as follows:

- **project_functions/**
  - *fake_data.py*: Module for generating fake data.
  - *geocoder_utils.py*: Utilities for geocoding.
  - *knapsack_problem.py*: Implementation of the knapsack problem.
  - *magical_functions.py*: Magical functions for special purposes.
  - *preprocessing.py*: Data preprocessing functions.
  - *tsp_solver.py*: Solver for the traveling salesman problem.
  - *visualization.py*: Functions for data visualization.
  - *__init__.py*: Initialization script for the package.

- **cleaning_data_with_incorrect_geodata.ipynb**: Jupyter Notebook for cleaning data with incorrect geodata.
- **main.ipynb**: Main Jupyter Notebook for the project.
- **README.md**: Project documentation.
- **config/**: Configuration files.
  - *additional_sources.txt*: Additional data sources.
  - *requirements.txt*: Project dependencies.

- **data/**
  - **data_after_geocoding/**
    - *geodata.csv*: Geocoded data.

  - **input_data/**
    - *czetotliwosci_one-hot-encoder.xlsx*: Encoder data.
    - *Fake_data.csv*: Fake data.
    - *odpady_kompresja.csv*: Compressed waste data.
    - *samochody.csv*: Vehicle data.
    - *samochody_perezap.csv*: Reloaded vehicle data.
    - *typy_pojemnik.csv*: Container types.
   
      
  - **sorted_by_groups/**
    - *1Cz_B_150102_150106_150104_200139.csv*: Route data for specific group.
    - ...

  - **routes/**
    - **genetic_groups_2024-01-13/**
      - *rout_1Cz_B_150101_200201.csv*: Route data.

    - **or_tools_groups_2024-01-13/**
      - *rout_1Cz_B_150101_200201.csv*: Route data.
      - ...

      - **map/**
        - *1Cz_B_150101_200201_map_rout_0.html*: Map for the route.
        - *1Cz_B_150102_150106_150104_200139_map_rout_0.html*: Map for the route.
        - ...

# Installation and Dependencies

## Installation

1. **Python:**
Ensure you have a version of Python installed that is compatible with the project (Python 3.x is recommended).

   Install the following libraries from the Python standard library:
   ```
   bash
   pip install faker tqdm keyboard
   ```
   
   Install the following libraries that require additional installationinstallation:
     
   ```
   bash
   pip install pandas numpy geopy scikit-learn ortools folium openrouteservice matplotlib seaborn
   ```
   
## Token Files:

Add the following token files to the `data` folder:

- `token_OpenRouteService.json`
- `token_pushbullet.json`

## Dependencies:

### From the Python standard library:
- `os`: Operating system-specific functionality (for example, reading or writing to the file system).
- `csv`: Read and write CSV files.
- `concurrent.futures`: Library for parallel execution of tasks.
- `requests`: Send HTTP requests to the server.
- `datetime`, `time`, `random`: Working with dates, times and generating random numbers.
- `math.ceil`: Round up.
- `itertools.product`: Combinatorial method for creating a product of iterable objects.
- `tqdm`, `IPython.display`: Progress bars and display in Jupyter Notebook.
- `keyboard`: Library for working with the keyboard.

### Libraries that require installation:
- `pandas`, `numpy`: Libraries for data manipulation.
- `geopy`: Calculate distances between geographic coordinates.
- `scikit-learn`: Implementation of machine learning algorithms, including KMeans and DBSCAN.
- `ortools`: Google OR-Tools Constraint Solver.
- `pickle`: Serialize and deserialize Python objects.
- `re`: Library for working with regular expressions.
- `folium`: Library for interactive maps.
- `openrouteservice`: Python client for the OpenRouteService API.
- `matplotlib`, `seaborn`: Libraries fUsage Exampleslizations.оздания визуализаций.
eyboar

## Examples of using

  You can see examples of use in the maijupyter notebook main.ipynbok.

## Contribution to the project and license

We welcome your contributions! If you have ideas, bug fixes or other suggestions, feel free to contribute to the project. To offer a contribution:

1. [Fork the project](https://github.com/your_user/your_project/fork).
2. Create your own branch (`git checkout -b your_branch`).
3. Make changes and submit them (`git commit -am 'New functionality added' Create a Pull Request.he changes to your fork (`git push origin your_branch`).
5. Create a Pull cense

This project is licensed under the [MIT](LICENSE ) License.Request.



## Contact Information

1. Mail: [Oleksandr Kanslosh](aleksandrkanalosh@gmail.com)
2. LinkedIn: [Oleksandr Kanslosh](https://www.linkedin.com/in/oleksandr-kanslosh)


## Plans

- [ ] Expand the functionality of the `DistanceMatrixCalculator` class to allow you to obtain the real distance between points.
- [ ] Create a new class to optimize trip frequency and find the best solutions.

