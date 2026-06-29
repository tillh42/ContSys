
import os 
from dotenv import load_dotenv
from edgeml import DatasetReceiver, Dataset
import pandas as pd


# Get API Keys
load_dotenv()
write_key = os.getenv("EDGE_WRITE_KEY")
read_key = os.getenv("EDGE_READ_KEY")

# Uses Read- and Write-Key!
receiver = DatasetReceiver(
    backendURL="https://app.edge-ml.org",
    readKey=read_key,
    writeKey=write_key
)

# Gather all Datasets that have a certain name
def gather(receiver: DatasetReceiver, name: str):
    return [ds for ds in receiver.datasets if getName(ds) == name]    

# Gather own Datasets that have a certain name
def gatherOwn(receiver: DatasetReceiver, name: str):
    return [ds for ds in getOwn(receiver=receiver) if getName(ds) == name]    

# Get the label of the dataset (e.g. "swimming")
def getName(ds: Dataset):
    if ds.labelings == []:
        return ""
    return ds.labelings[0].labels[-1].name.lower()

# Get the dataset that I have created
def getOwn(receiver: DatasetReceiver):
    return [ds for ds in receiver.datasets if ds.name.startswith("uoihp")]    

# Learning Labels (datasets with labels)
sitting_ds = gatherOwn(receiver=receiver, name="sitting")
walking_ds = gatherOwn(receiver=receiver, name="walking")
stairs_ds  = gatherOwn(receiver=receiver, name="stairs")

# Debug
print("Number of sitting Datasets:\t", len(sitting_ds))
print("Number of walking Datasets:\t", len(walking_ds))
print("Number of stairs  Datasets:\t", len(stairs_ds))

# Important columns: time accX accY accZ
# activity/context is already filtered above
# Example:
#                        time  accX  accY  accZ activity
# 0   2026-05-10 10:40:42.596  -5.0   NaN   NaN  sitting
# 1   2026-05-10 10:40:42.597   NaN   2.4   8.0  sitting
# 2   2026-05-10 10:40:42.615  -5.0   NaN   NaN  sitting
# 3   2026-05-10 10:40:42.616   NaN   2.4   7.9  sitting
# 4   2026-05-10 10:40:42.628  -5.0   2.5   7.9  sitting
# ..                      ...   ...   ...   ...      ...

# Resample data because of different sample rates
# Convert datasets into one dataframe per activity
def datasets_to_df(datasets, activity):
    frames = []

    for ds in datasets:
        ds.loadData()
        df = ds.data.copy()  # adjust if your API uses another attribute

        # Keep only relevant columns
        df = df[["time", "accX", "accY", "accZ"]]

        # Convert timestamp
        df["time"] = pd.to_datetime(df["time"])

        # Add label
        df["activity"] = activity

        frames.append(df)

    return pd.concat(frames, ignore_index=True)

def datasets_to_df_with_trimmed_edges(datasets, activity):
    frames = []

    for ds in datasets:
        ds.loadData()
        df = ds.data.copy()

        df = df[["time", "accX", "accY", "accZ"]]
        df["time"] = pd.to_datetime(df["time"])
        
        # --- NEW: TRIM THE POCKET TRANSITIONS ---
        # Sort by time to be absolutely sure
        df = df.sort_values("time")
        
        # Find the timestamp thresholds
        start_time = df["time"].min() + pd.Timedelta(seconds=2)
        end_time = df["time"].max() - pd.Timedelta(seconds=2)
        
        # Keep only the data in the middle
        df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]
        # ----------------------------------------

        df["activity"] = activity
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


# trimm the first seconds because the movement of phnone into/out of pocket is also recorded
sitting_df = datasets_to_df_with_trimmed_edges(sitting_ds, "sitting")
walking_df = datasets_to_df_with_trimmed_edges(walking_ds, "walking")
stairs_df  = datasets_to_df_with_trimmed_edges(stairs_ds, "stairs")

# One big dataframe
df = pd.concat(
    [sitting_df, walking_df, stairs_df],
    ignore_index=True
)

# 1. Set the time as the index and sort it chronologically
df_sorted = df.set_index("time").sort_index()

# 2. Group by activity, apply the rolling window, and aggregate
df_windowed = (
    df_sorted.groupby("activity")
      .rolling(window="1s")  # 'on="time"' is no longer needed since it's the index
      .agg(["mean", "var", "std"])
      .dropna()
      .reset_index()          # Brings 'activity' and 'time' back as normal columns
)

# 3. Clean up MultiIndex column names (e.g., 'accX_mean')
df_windowed.columns = [
    f"{col[0]}_{col[1]}" if isinstance(col, tuple) and col[1] else col[0]
    for col in df_windowed.columns
]

print(df_windowed)

import matplotlib.pyplot as plt
import seaborn as sns

# 1. Melt the dataframe to long-form
df_melted = df_windowed.melt(
    id_vars=["time", "activity"],
    value_vars=[
        "accX_mean", "accX_var", "accX_std",
        "accY_mean", "accY_var", "accY_std",
        "accZ_mean", "accZ_var", "accZ_std"
    ],
    var_name="Sensor_Metric",
    value_name="Value"
)

# 2. Split the names into separate 'Axis' and 'Metric' columns
df_melted["Axis"] = df_melted["Sensor_Metric"].apply(lambda x: x.split("_")[0].replace("acc", ""))
df_melted["Metric"] = df_melted["Sensor_Metric"].apply(lambda x: x.split("_")[1])

# 3. Set style
sns.set_theme(style="darkgrid")

# 4. Create a 3x3 grid of completely independent plots
g = sns.relplot(
    data=df_melted,
    x="time",
    y="Value",
    hue="activity",    # Colors represent sitting, walking, stairs
    row="Metric",      # Row 1 = mean, Row 2 = var, Row 3 = std
    col="Axis",        # Col 1 = X, Col 2 = Y, Col 3 = Z
    kind="line",
    height=3,          # Height of each individual subplot
    aspect=1.5,        # Width of each individual subplot
    facet_kws={"sharey": False, "sharex": True} # Independent Y scales, shared Time axis
)

# 5. Clean up headers and format dates
g.set_titles(template="{row_name} | Axis {col_name}")
g.figure.autofmt_xdate()

plt.tight_layout()

plt.savefig("feature_plot")
