# Setup
import pandas as pd
noaa_fire_data_df = pd.read_csv('../input/noaa_wildfires.csv')
noaa_fire_first_quarter = noaa_fire_data_df[(noaa_fire_data_df['start_day_of_year'] <= 92) & (noaa_fire_data_df['cause'] != 'U')] # Filters out unknown causes
regions = ['California', 'Great Basin', 'Northwest', 'Inland Empire', 'Rocky Mountains', 'Southwest']
breakdowns = []
for region in regions:
    print(f"Investigating {region}...")
    region_df = noaa_fire_first_quarter[noaa_fire_first_quarter['region'] == region]
    print(len(region_df))
    natural_percentage = (region_df['cause'] == 'N').mean() * 100
    lightning_percentage = (region_df['cause'] == 'L').mean() * 100
    human_percentage = (region_df['cause'] == 'H').mean() * 100
    other_percentage = (region_df['cause'] == 'O').mean() * 100
    print(natural_percentage, lightning_percentage, human_percentage, other_percentage)