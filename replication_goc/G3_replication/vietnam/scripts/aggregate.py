import warnings; warnings.filterwarnings("ignore")
import glob, numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.windows import from_bounds
from rasterstats import zonal_stats
g = pd.read_pickle("districts_geo.pkl"); g = gpd.GeoDataFrame(g, geometry="geometry", crs=4326)
for y in (2010, 2020):
    tif = glob.glob(f"ghs{y}/*.tif")[0]
    with rasterio.open(tif) as src:
        win = from_bounds(101.5, 7.5, 110.5, 24.0, src.transform)
        arr = src.read(1, window=win).astype("float64"); tr = src.window_transform(win)
        nod = src.nodata
    if nod is not None: arr[arr == nod] = 0
    zs = zonal_stats(g.geometry, arr, affine=tr, stats=["sum", "count"], nodata=-1, all_touched=False)
    g[f"built_m2_{y}"] = [z["sum"] if z["sum"] is not None else 0 for z in zs]
    print(y, "cells per district median", np.median([z["count"] for z in zs]))
rwi = pd.read_csv("vnm_rwi.csv")
pts = gpd.GeoDataFrame(rwi, geometry=gpd.points_from_xy(rwi.longitude, rwi.latitude), crs=4326)
jj = gpd.sjoin(pts, g[["wiki_idx", "geometry"]], predicate="within")
agg = jj.groupby("wiki_idx").agg(rwi_mean=("rwi", "mean"), rwi_n=("rwi", "size"), rwi_err_mean=("error", "mean"))
g = g.merge(agg, left_on="wiki_idx", right_index=True, how="left")
g["built_share_2010"] = g.built_m2_2010 / (g.area_geo * 1e6)
g["built_share_2020"] = g.built_m2_2020 / (g.area_geo * 1e6)
g.to_pickle("districts_full.pkl")
print(g[["name", "province", "popn", "area_km2", "built_share_2020", "built_share_2010", "rwi_mean", "rwi_n"]].describe().round(3).to_string())
print("missing rwi:", g.rwi_mean.isna().sum())
