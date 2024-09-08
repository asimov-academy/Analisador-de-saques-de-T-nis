
import pickle
import os
import pandas as pd
import pdb
import numpy as np

folder = "landmarks"

for file in [i for i in os.listdir(folder) if ('.' != i[0]) and ('norm' not in i)]:
    land_file = os.path.join(folder, file)

    with open(land_file, 'rb') as f:
        landmarks_data = pickle.load(f)

        landmarks = []
        for frame in landmarks_data:
            frame_array = []
            for point in frame.landmark:
                frame_array.append([point.x, point.y, point.z])
            # pdb.set_trace()
            df = pd.DataFrame(frame_array, columns=["x", "y", "z"])

            df['y'] = -df['y']
            x_scale = (df["y"][0] - df["y"].min()) / (df["x"].max() - df["x"].min())
            df['x'] = (df["x"] - df["x"].min()) / (df["x"].max() - df["x"].min()) / x_scale
            df['y'] = (df["y"] - df["y"].min()) / (df["y"][0] - df["y"].min())
            df['y'] = (df["y"] - df["y"].min()) / (df["y"].max() - df["y"].min())
            df['z'] = (df["z"] - df["z"].min()) / (df["z"].max() - df["z"].min())
            landmarks.append(df)

    norm_file = "norm_" + file
    norm_file = os.path.join(folder, norm_file)
    with open(norm_file, 'wb') as f:
        print(f"Saving {norm_file}")
        pickle.dump(landmarks, f)
