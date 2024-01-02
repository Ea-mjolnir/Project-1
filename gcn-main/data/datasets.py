import numpy as np
import pandas as pd
import torch
import pyproj
from torch_geometric.data import Dataset, Data
import torch_geometric.utils as tgutils
import os
import ast

class BuildingFootprintsDataset(Dataset):
    def __init__(self, csv_path=os.path.dirname(__file__)+'/building_data.csv', transform=None, pre_transform=None, centroid=True):
        super(BuildingFootprintsDataset, self).__init__(transform, pre_transform)
        self.data_df = pd.read_csv(csv_path)
        self.proj = pyproj.Transformer.from_crs(4326, 3857)
        self.centroid = centroid

    def len(self):
        return len(self.data_df['PolygonID'].unique())

    def get(self, idx):
        # Extract polygon vertices from csv (grouped by polygon id)
        polygon_id = self.data_df['PolygonID'].unique()[idx]
        vertices_df = self.data_df[self.data_df['PolygonID'] == polygon_id]

        # Sort vertices based on direction_sort column
        vertices_df = vertices_df.sort_values(by='direction_sort')

        # Convest lat-lon crs in degrees to a crs with meter unit
        vertices_df['coord_meter'] = vertices_df[['X', 'Y']].apply(lambda p: self.proj.transform(p.X, p.Y), axis=1)
        vertices_df[['X_meter','Y_meter']] = vertices_df['coord_meter'].apply(lambda p: pd.Series(np.array([p[0], p[1]]), index=["X_meter", "Y_meter"]))

        if self.centroid:
            # For each polygon, translate its origin(0,0) to its centroid
            cen_X = vertices_df['X_meter'].mean()
            cen_Y = vertices_df['Y_meter'].mean()
            vertices_df['X_local'] = vertices_df['X_meter'].apply(lambda x: x - cen_X)
            vertices_df['Y_local'] = vertices_df['Y_meter'].apply(lambda y: y - cen_Y)
        else:
            # translate so that all coordinates remain positive (origin bottom-left instead of centroid)
            min_X = vertices_df['X_meter'].min()
            min_Y = vertices_df['Y_meter'].min()
            vertices_df['X_local'] = vertices_df['X_meter'] - min_X
            vertices_df['Y_local'] = vertices_df['Y_meter'] - min_Y

        vertices = vertices_df[['X_local','Y_local']].values
        labels = vertices_df['is_corner'].values

        # Create nodes and labels tensor
        node_features = torch.tensor(vertices, dtype=torch.float32) # use dtype=torch.float64 for increased precision
        labels = torch.tensor(labels, dtype=torch.long) # has to be dtype=torch.long for some reason

        # Create edges
        num_vertices = len(vertices)
        edge_indices = []
        for i in range(num_vertices):
            edge_indices.append([i, (i + 1) % num_vertices])  # Connect neighboring vertices
            edge_indices.append([(i + 1) % num_vertices, i])  # Add reverse direction, to get an undirected graph
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # edge_indices = tgutils.add_self_loops(edge_indices)[0]

        # create graph structure
        data = Data(x=node_features, edge_index=edge_indices, y=labels)

        return data



class SimplificationDataset(Dataset):
    def __init__(self, csv_path=os.path.dirname(__file__) + '/buildings_simpl_action.csv', include_features=False):
        super(SimplificationDataset, self).__init__()
        self.proj = pyproj.Transformer.from_crs(4326, 3857)
        self.include_features = include_features
        self.data_df = pd.read_csv(csv_path)

    def len(self):
        return len(self.data_df['polygon_id'].unique())

    def get(self, idx):
        # Extract polygon vertices from csv (grouped by polygon id)
        polygon_id = self.data_df['polygon_id'].unique()[idx]
        vertices_df = self.data_df[self.data_df['polygon_id'] == polygon_id].copy() # use copy() to fix SettingWithCopyWarning

        # translate so that all coordinates remain positive
        min_X = vertices_df['x'].min()
        min_Y = vertices_df['y'].min()
        vertices_df['x_local'] = vertices_df['x'] - min_X
        vertices_df['y_local'] = vertices_df['y'] - min_Y

        vertices = vertices_df[['x_local','y_local']].values
        # labels = vertices_df['action_type'].values
        
        vertices_df.loc[vertices_df['action_type'].ne(1),'move_position'] = "(0.0, 0.0)"
        vertices_df[['move_x','move_y']] = vertices_df['move_position'].apply(lambda pos_str: pd.Series(ast.literal_eval(pos_str), index=["move_x", "move_y"]))
        vertices_df['move_x'][vertices_df['action_type'] == 1] = vertices_df['move_x'][vertices_df['action_type'] == 1] - min_X
        vertices_df['move_y'][vertices_df['action_type'] == 1] = vertices_df['move_y'][vertices_df['action_type'] == 1] - min_Y
        
        vertices_df['move_res_x'] = vertices_df['move_x'] - vertices_df['x_local']
        vertices_df['move_res_y'] = vertices_df['move_y'] - vertices_df['y_local']
        vertices_df.loc[vertices_df['action_type'].ne(1),'move_res_x':'move_res_y'] = 0.0
        
        # if idx == 0:
        #     print(vertices_df)
        
        if self.include_features:
            vertices = vertices_df[['x_local', 'y_local', 'turning_angle']].values
        else:
            vertices = vertices_df[['x_local','y_local']].values

        labels = vertices_df[['action_type','move_res_x','move_res_y']].values

        # Create nodes and labels tensor
        node_features = torch.tensor(vertices, dtype=torch.float32) # use dtype=torch.float64 for increased precision
        labels = torch.tensor(labels, dtype=torch.float32) # has to be dtype=torch.long for some reason

        # Create edges
        num_vertices = len(vertices)
        edge_indices = []
        for i in range(num_vertices):
            edge_indices.append([i, (i + 1) % num_vertices])  # Connect neighboring vertices
            edge_indices.append([(i + 1) % num_vertices, i])  # Add reverse direction, to get an undirected graph
        edge_indices = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

        # create graph structure
        data = Data(
            x=node_features, 
            edge_index=edge_indices, 
            y=labels, 
            pos=torch.tensor(vertices_df[['x','y']].values, dtype=torch.float64), 
            poly_id=polygon_id
        )

        return data