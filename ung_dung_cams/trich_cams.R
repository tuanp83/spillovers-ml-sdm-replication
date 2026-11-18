# Trích CAMS EAC4 (0,75°) trung bình năm 2020 cho 638 huyện G3 (trung bình có trọng số diện tích).
suppressMessages({ library(terra); library(data.table) })
f <- path.expand("~/NCKH/Kinhteyte_13_11_2026-T.Minh/du_lieu_chung/cams/eac4/don_tang/data_allhours_sfc.nc")
h <- vect(path.expand("~/NCKH/CNN-SDM_17.11.2026/du_lieu_o_nhiem/ghep_huyen/huyen_G3_638.gpkg")); h <- h[order(h$g3_row)]
V <- c("pm2p5", "aod550", "bcaod550", "omaod550", "suaod550", "duaod550", "tcno2", "tcco", "tcso2", "tchcho")
yr <- 2005 + (0:227) %/% 12
out <- data.table(g3_row = h$g3_row)
for (v in V) { r <- rast(paste0("NETCDF:\"", f, "\":", v)); stopifnot(nlyr(r) == 228)
  m <- mean(r[[yr == 2020]]); out[[v]] <- extract(m, h, fun = mean, exact = TRUE, ID = FALSE)[, 1] }
r <- rast(paste0("NETCDF:\"", f, "\":pm2p5"))[[1]]
out$cell <- cellFromXY(r, crds(centroids(h, inside = TRUE)))
fwrite(out, "cams_huyen_2020.csv"); cat("ô khác nhau:", length(unique(out$cell)), " NA:", sum(is.na(out)), "\n")
