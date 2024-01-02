import geopandas as gpd
from geopy.distance import distance
import pandas as pd

# create midpoint
def calculate_midpoint(lat1, lon1, lat2, lon2, m1, m2):
    final_x ,final_y, final_direction = ((lon1+lon2)/2), ((lat1+lat2)/2), ((m1+m2)/2)
    return final_x, final_y, final_direction

# creates two points in between
def create_two_intermediate_points(lat1, lon1, lat2, lon2, m1, m2):
    int1_point_lat, int1_point_lon, int1_m = (lat1 + (lat2 - lat1) / 3, lon1 + (lon2 - lon1) / 3, (m1+m2)/2-0.1)
    int2_point_lat, int2_point_lon, int2_m = (lat1 + 2 * (lat2 - lat1) / 3, lon1 + 2 * (lon2 - lon1) / 3, (m1+m2)/2+0.1)
    return int1_point_lat, int1_point_lon, int1_m, int2_point_lat, int2_point_lon, int2_m

# create three points in between
def create_three_intermediate_points(lat1, lon1, lat2, lon2, m1, m2):
    int1_point_lat, int1_point_lon, int1_m = (lat1 + (lat2 - lat1) / 4, lon1 + (lon2 - lon1) / 4, (m1+m2)/2-0.2)
    int2_point_lat, int2_point_lon, int2_m = (lat1 + 2 * (lat2 - lat1) / 4, lon1 + 2 * (lon2 - lon1) / 4, (m1+m2)/2 )
    int3_point_lat, int3_point_lon, int3_m = (lat1 + 3 * (lat2 - lat1) / 4, lon1 + 3 * (lon2 - lon1) / 4, (m1+m2)/2+0.2)
    return int1_point_lat, int1_point_lon, int1_m, int2_point_lat, int2_point_lon, int2_m, int3_point_lat, int3_point_lon, int3_m

# calculate distance
def calculate_distance(lat1, lon1, lat2, lon2):
    point1 = (lat1, lon1)
    point2 = (lat2, lon2)
    dist = distance(point1, point2).meters
    return dist

# Read the shapefile
shapefile_path = "buildings_20k_cleaned/buildings_20k_cleaned.shp"
data = gpd.read_file(shapefile_path)

# Extract vertices of polygons and mark them as 1, add direction
vertices_list = []
for i, geometry in enumerate(data['geometry']):
    m = 9
    if geometry.geom_type == 'Polygon':
        polygon_id = i + 1
        vertices = list(geometry.exterior.coords)
        for vertex in vertices:
            x, y = vertex
            vertices_list.append((polygon_id, x, y, "1", m+1))
            m = m+10

for i in range(len(vertices_list)):
    if vertices_list[i][0] == vertices_list[i+1][0]:
        lat1, lon1, lat2, lon2 = vertices_list[i][2], vertices_list[i][1], vertices_list[i+1][2], vertices_list[i+1][1]
        m1, m2 = vertices_list[i][4], vertices_list[i+1][4]

        if calculate_distance(lat1, lon1, lat2, lon2) <= 2:
            mid_x, mid_y, mid_m = calculate_midpoint(lat1, lon1, lat2, lon2,m1,m2)
            vertices_list.append((vertices_list[i][0], mid_x, mid_y, "0", mid_m))
            
        elif calculate_distance(lat1, lon1, lat2, lon2) <= 10 and 2 < calculate_distance(lat1, lon1, lat2, lon2):
            int1_point_lat, int1_point_lon, int1_m, int2_point_lat, int2_point_lon, int2_m = create_two_intermediate_points(lat1, lon1, lat2, lon2,m1,m2)            
            vertices_list.append((vertices_list[i][0], int1_point_lon, int1_point_lat, "0", int1_m))
            vertices_list.append((vertices_list[i][0], int2_point_lon, int2_point_lat, "0", int2_m))

        elif calculate_distance(lat1, lon1, lat2, lon2) > 10:
            int1_point_lat, int1_point_lon, int1_m, int2_point_lat, int2_point_lon, int2_m, int3_point_lat, int3_point_lon, int3_m = create_three_intermediate_points(lat1, lon1, lat2, lon2, m1, m2)
            vertices_list.append((vertices_list[i][0], int1_point_lon, int1_point_lat, "0", int1_m))
            vertices_list.append((vertices_list[i][0], int2_point_lon, int2_point_lat, "0", int2_m))
            vertices_list.append((vertices_list[i][0], int3_point_lon, int3_point_lat, "0", int3_m))

df = pd.DataFrame(vertices_list, columns=['PolygonID', 'X', 'Y', 'is_corner', 'direction_sort'])

# Sort the DataFrame by 'direction_sort' in descending order within each 'polygonid'
df_sorted = df.sort_values(by='direction_sort', ascending=True)

# Drop duplicates, keeping the first occurrence (highest 'direction_sort') for each 'polygonid'
filtered_data = df_sorted.groupby('PolygonID').apply(lambda x: x.iloc[:-1]).reset_index(drop=True)

filtered_data.to_csv('output2.csv', index=False)

print("CSV file saved successfully.")