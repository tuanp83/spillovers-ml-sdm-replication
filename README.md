# Replication files for "Spillovers Manufactured by Machine Learning: Estimation and Inference with Predicted Regressors in Spatial Durbin Models"

Nguyen Khac Minh and Pham Anh Tuan.

## Requirements
- Python 3.9+ with `numpy`, `scipy`, `pandas`, `geopandas`, `unidecode` and `pypdf`.
- R 4.x with `terra` and `data.table`, needed only to rebuild the pollution panels.

Set the environment variable `G3_ROOT` to this folder (the default is the current folder) and run the scripts from it. Most scripts accept the number of replications as their first argument.

## Core code (`g3_code/`)
- `g3_v2.py`: LC-IV estimator, design variance, and the subvector AR statistics $T_\theta$, $T_\rho$, $T_{ATI}$ (Theorems 2–3).
- `g3_v2_ident.py`: $T_{id}$ and population identification strength (Proposition 3).
- `g3_sigma_h.py`: feasible tests with the capped variance at the hypothesis, including closed-form ray limits (Theorem 4, Remark 4).
- `g3_design.py`: Horvitz–Thompson LC-IV and design variances for stratified and multistage audits (Proposition 2).

## Tables and figures
| Output | Script |
|---|---|
| Tables on the sign law, point estimation, CNN experiments, contaminated images; Figure S1 | `replication_goc/G3_replication/code/` (`g3_sim.py`, `g3_fix.py`, `g3_cnn*.py`, `g3_a1.py`, `g3_scaling2.py`) |
| Table on size of subvector AR tests (old vs new variance) | `ung_dung_sigma/muc7/mc_muc7.py size 2000` |
| Table on length of confidence sets | `ung_dung_sigma/muc7/mc_muc7.py len 300` |
| Table on informativeness of $CS_\rho$ | `replication_goc/G3_replication/code/g3_v2_ident.py` |
| Table on the feasible variance; numbers on distant alternatives | `ung_dung_sigma/mc_sigma.py 1000`, `ung_dung_sigma/kiem_tra_tran.py` |
| Smooth-regressor and spatially dependent shock checks | `ban_thao_v4/kiem_chung/phan_vi_du.py`, `ban_thao_v6/kiem_chung/chk_smooth_new.py`, `ban_thao_v6/kiem_chung/chk_nonid_n.py` |
| Online Supplement S3 (complex audit designs) | `ung_dung_thiet_ke/mc_thiet_ke.py 1000 0.2 GHSL` |
| Numerical check after Proposition 1 (Vietnamese districts) | `ung_dung_cams/kiem_chung_menh_de.py` |
| Vietnamese illustration: RWI and semi-synthetic SDM | `replication_goc/G3_replication/vietnam/scripts/` |
| Vietnamese illustration: emissions outcome | `ung_dung_phat_thai/phan_tich.py`, `ung_dung_phat_thai/bo_sung.py` |
| Vietnamese illustration: coarse CAMS predictor | `ung_dung_cams/vn_cams.py`, `ung_dung_cams/tach_co_che.py` |
| Population validation against official statistics | `du_lieu_dan_so/doi_chieu_gso.py`, `du_lieu_dan_so/vung_gso.py` |

## Data
- **Districts and weights:**
  - `du_lieu_o_nhiem/ghep_huyen/vn_districts_G3.csv` contains the 638 districts: population and area, GHSL built-up shares, Meta RWI, and the training flag.
  - `W_queen_rowstd.npz` is the row-standardised queen-contiguity matrix.
- **Emissions:** `du_lieu_o_nhiem/phat_thai_huyen_G3_rong.csv` holds district-level CAMS-GLOB-ANT v6.2 emissions, built with `du_lieu_o_nhiem/tong_hop_huyen.R` from files downloaded from the Copernicus Atmosphere Data Store.
- **Reanalysis:** `ung_dung_cams/cams_huyen_2020.csv` holds district-level CAMS reanalysis (EAC4) averages for 2020.
- **Population source and validation:**
  - District populations are taken from the government's administrative-merger dossiers (`du_lieu_dan_so/districts_wiki_raw.csv`, transcribed from a public compilation).
  - They are validated against General Statistics Office (2022), Table 1, via `du_lieu_dan_so/doi_chieu_gso.py`. The source PDF is publicly available from nso.gov.vn and is not redistributed here.
- **Not redistributed:** raw satellite and reanalysis grids (GHSL, CAMS). They are publicly available from the providers cited in the article.

Some comments in the code are in Vietnamese.
