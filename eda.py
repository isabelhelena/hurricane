import os
import pandas as pd
import matplotlib.pyplot as plt

DATA = "data/sample_hurdat2_subset.csv"
FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)

df = pd.read_csv(DATA, parse_dates=["datetime"])

plt.hist(df["wind"], bins=20)
plt.title("Wind Speed Distribution (knots)")
plt.savefig(f"{FIG_DIR}/wind_hist.png")

plt.clf()
plt.hist(df["pressure"], bins=20)
plt.title("Central Pressure Distribution (mb)")
plt.savefig(f"{FIG_DIR}/pressure_hist.png")

plt.clf()
for sid, grp in df.groupby("storm_id"):
    plt.plot(grp["lon"], grp["lat"], marker="o")
plt.title("Storm Tracks (Lon/Lat)")
plt.savefig(f"{FIG_DIR}/tracks_scatter.png")

print("EDA figures saved to", FIG_DIR)
