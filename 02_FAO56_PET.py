import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = "/Users/vikrantjamwal/Downloads/ERA5_Hourly_2024.csv"
df = pd.read_csv(file, parse_dates=["time"])

expected = ["time","t2m","d2m","u10","v10","Rn"]
missing = [c for c in expected if c not in df.columns]
if missing:
    raise ValueError(f"Missing columns in CSV: {missing}")

df["date"] = df["time"].dt.date
daily = (df.groupby("date").agg({"t2m":"mean","d2m":"mean","u10":"mean",
                                 "v10":"mean","Rn":"sum"}).reset_index())

lat, alt, z = 31.7087, 780, 10
Gsc, sigma = 0.0820, 4.903e-9

wind_adj_factor = 4.87 / np.log(67.8*z - 5.42)
daily["u2"] = np.sqrt(daily["u10"]**2 + daily["v10"]**2) * wind_adj_factor

daily["es"] = 0.6108*np.exp(17.27*daily["t2m"]/(daily["t2m"]+237.3))
daily["ea"] = 0.6108*np.exp(17.27*daily["d2m"]/(daily["d2m"]+237.3))
daily["delta"] = 4098*daily["es"]/(daily["t2m"]+237.3)**2

P = 101.3*((293-0.0065*alt)/293)**5.26
gamma = 0.000665*P

G = 0
num = (0.408*daily["delta"]*(daily["Rn"]-G) +
       gamma*(900/(daily["t2m"]+273))*daily["u2"]*(daily["es"]-daily["ea"]))
den = daily["delta"] + gamma*(1+0.34*daily["u2"])
daily["PET_mm_day"] = num/den

plt.figure(figsize=(10,5))
plt.plot(pd.to_datetime(daily["date"]),daily["PET_mm_day"],lw=1.2,color="steelblue")
plt.title("Daily Potential Evapotranspiration (FAO-56 Penman-Monteith, 2024)")
plt.xlabel("Date (2024)"); plt.ylabel("PET (mm/day)")
plt.grid(True,linestyle="--",alpha=0.6); plt.tight_layout(); plt.show()

daily["month"] = pd.to_datetime(daily["date"]).dt.month_name()
monthly = daily.groupby("month")["PET_mm_day"].mean().sort_values(ascending=False)
max_month,max_value = monthly.idxmax(),monthly.max()
print(f"Highest average PET in: {max_month} ({max_value:.2f} mm/day)")

out_csv = "/Users/vikrantjamwal/Downloads/ERA5_daily_PET_2024.csv"
daily.to_csv(out_csv,index=False)
print(f"Daily PET saved -> {out_csv}")
