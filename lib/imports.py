# Libraries available in the Python standard library
import os  # Operating system-dependent functionality (e.g., reading or writing to the file system)
import csv  # Reading and writing CSV files
import json  # JavaScript Object Notation (JSON) data interchange format
import concurrent.futures  # Library for parallel execution of tasks
import requests  # Sending HTTP requests to a server
from datetime import datetime  # Library for working with dates and times
import time  # Library for working with time
import random  # Library for generating random numbers
from math import ceil  # Function for rounding up
from itertools import product  # Combinatorial method for creating a product of iterable objects
from tqdm.notebook import tqdm  # Progress bars for loops in Jupyter Notebooks
from IPython.display import display, clear_output, IFrame  # Displaying content in Jupyter Notebooks
import keyboard  # Library for working with the keyboard
from faker import Faker  # Library for generating fake data

# Libraries requiring installation
import pandas as pd  # Data manipulation library (abbreviated as pd conventionally)
import numpy as np  # Numerical computing library (abbreviated as np conventionally)
from geopy.distance import geodesic  # Calculating distances between geographic coordinates
from sklearn.cluster import KMeans  # K-Means clustering algorithm
from sklearn.cluster import DBSCAN  # Density-Based Spatial Clustering of Applications with Noise
from sklearn.preprocessing import StandardScaler  # Standardizing features by removing the mean and scaling to unit variance

# Solver
import array
from functools import partial
from deap import base, creator, tools, algorithms
from deap.algorithms import varAnd

from ortools.constraint_solver import pywrapcp  # Google OR-Tools Constraint Solver
from ortools.constraint_solver import routing_enums_pb2  # Enums for Google OR-Tools Constraint 

import pickle  # Python object serialization and deserialization
from typing import Union, Tuple  # Type hints for function arguments and return values

import re  # Regular expressions library

import folium  # Interactive maps library
import openrouteservice  # Python client for the OpenRouteService API
from openrouteservice.exceptions import ApiError  # Exception handling for OpenRouteService API

import matplotlib.pyplot as plt  # Plotting library for creating static, animated, and interactive visualizations
import seaborn as sns  # Statistical data visualization library
