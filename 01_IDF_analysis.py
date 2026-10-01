import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r, lognorm, kstest
from scipy.optimize import minimize
import matplotlib.ticker as mticker

csv_file = "/Users/vikrantjamwal/Desktop/pgrib/ERA5_precip_31.7087_76.9320.csv"
df = pd.read_csv(csv_file)

date_col = next((c for c in df.columns if "date" in c.lower()), None)
time_col = next((c for c in df.columns if "time" in c.lower()), None)
precip_col = next((c for c in df.columns if "precip" in c.lower()), None)
lat_col = next((c for c in df.columns if "lat" in c.lower()), None)
lon_col = next((c for c in df.columns if "lon" in c.lower()), None)

if date_col and time_col:
    df["time"] = pd.to_datetime(df[date_col].astype(str) + " " + df[time_col].astype(str))
elif time_col:
    df["time"] = pd.to_datetime(df[time_col])
else:
    raise KeyError("Could not find date/time columns.")

df = df.rename(columns={lat_col:"latitude", lon_col:"longitude", precip_col:"precip_mm"})
df = df[["time","latitude","longitude","precip_mm"]].sort_values("time")

dt_hr = df["time"].diff().dt.total_seconds().dropna().median() / 3600
if not np.isfinite(dt_hr) or dt_hr <= 0:
    dt_hr = 1.0

durations_hr, Tvals = [1,3,6,12,24], [2,5,10,25,50]
df["year"] = df["time"].dt.year
annual_max = {}

for D in durations_hr:
    w = int(round(D / dt_hr))
    df[f"sum_{D}h"] = df["precip_mm"].rolling(w, min_periods=w).sum()
    annual_max[D] = df.groupby("year")[f"sum_{D}h"].max().dropna()

def gumbel_icdf(p, mu, beta):
    return mu - beta * np.log(-np.log(p))

IDF_depths = np.zeros((len(Tvals), len(durations_hr)))

for j, D in enumerate(durations_hr):
    data = annual_max[D].values
    data = data[np.isfinite(data) & (data > 0)]
    shape, loc, scale = lognorm.fit(data, floc=0)
    KS_ln = kstest(data, "lognorm", args=(shape,loc,scale)).statistic
    mu, beta = gumbel_r.fit(data)
    KS_gum = kstest(data, "gumbel_r", args=(mu,beta)).statistic
    p = 1 - 1 / np.array(Tvals)
    if KS_ln < KS_gum:
        best, q = "Lognormal", lognorm.ppf(p, shape, loc=loc, scale=scale)
    else:
        best, q = "Gumbel", gumbel_icdf(p, mu, beta)
    IDF_depths[:,j] = q
    print(f"{D:2d} hr -> {best} (KS_ln={KS_ln:.4f}, KS_gum={KS_gum:.4f})")

IDF_intensity = IDF_depths / np.array(durations_hr)
IDF_table = pd.DataFrame(IDF_intensity, index=[f"T={T}yr" for T in Tvals],
                         columns=[f"{D}h" for D in durations_hr])
print(IDF_table.round(2))

D_flat, T_flat = np.meshgrid(durations_hr, Tvals)
I_all, T_all, D_all = IDF_intensity.ravel(), T_flat.ravel(), D_flat.ravel()
mask = np.isfinite(I_all) & (I_all > 0)
I_all, T_all, D_all = I_all[mask], T_all[mask], D_all[mask]

def idf_model(p,T,D):
    K,x,a,n = p
    return K*T**x/(D+a)**n

res = minimize(lambda p: np.sum((I_all-idf_model(p,T_all,D_all))**2),
               [100,0.2,5,0.7], method="Nelder-Mead")
K,x,a,n = res.x
I_pred = idf_model(res.x,T_all,D_all)
R2 = 1 - np.sum((I_all-I_pred)**2)/np.sum((I_all-I_all.mean())**2)

print(f"i = {K:.3f} * T^{x:.3f} / (D + {a:.3f})^{n:.3f}")
print(f"R² = {R2:.4f}")

plt.figure(figsize=(8,5))
for j,D in enumerate(durations_hr):
    plt.plot(Tvals,IDF_intensity[:,j],"o-",label=f"{D} hr")
plt.xscale("log"); plt.xticks(Tvals)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xlabel("Return Period (years)"); plt.ylabel("Rainfall Intensity (mm/hr)")
plt.title("IDF Curves - ERA5 Rainfall (1995-2024)")
plt.grid(True,which="both",ls="--",alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
colors = plt.cm.viridis(np.linspace(0,1,len(durations_hr)))
for j,D in enumerate(durations_hr):
    plt.plot(Tvals,IDF_intensity[:,j],"o",color=colors[j])
    plt.plot(Tvals,idf_model(res.x,np.array(Tvals),D),"--",color=colors[j],label=f"{D} hr")
plt.xscale("log"); plt.xticks(Tvals)
plt.gca().xaxis.set_major_formatter(mticker.ScalarFormatter())
plt.xlabel("Return Period (years)"); plt.ylabel("Intensity (mm/hr)")
plt.title("Observed vs Fitted IDF Curves")
plt.grid(True,which="both",ls="--",alpha=0.5); plt.legend(); plt.tight_layout(); plt.show()

idf_csv = "/Users/vikrantjamwal/Downloads/IDF_results.csv"
params_txt = "/Users/vikrantjamwal/Downloads/IDF_parameters.txt"
IDF_table.to_csv(idf_csv)
with open(params_txt,"w") as f:
    f.write(f"K = {K:.4f}\nx = {x:.4f}\na = {a:.4f}\nn = {n:.4f}\nR2 = {R2:.4f}\n")
print("Results saved.")
