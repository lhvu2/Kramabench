import kagglehub

# Download latest version
path = kagglehub.dataset_download("sobhanmoosavi/us-weather-events")
print("Path to us weather events dataset files:", path)
path_2 = kagglehub.dataset_download("robikscube/zillow-home-value-index")
print("Path to zillow home index value dataset files:", path_2)