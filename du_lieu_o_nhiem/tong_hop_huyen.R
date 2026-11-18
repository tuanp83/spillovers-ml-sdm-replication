# Tổng hợp CAMS-GLOB-ANT v6.2 (0,1°, đã cắt về VN: cams_glob_ant/vn/<biến>_<năm>.tif) lên 638 huyện G3.
# Mỗi huyện: trung bình có trọng số diện tích (exact) của thông lượng phát thải, quy về tấn/km²/năm,
# trung bình 12 tháng có trọng số số ngày. Nhóm ngành theo CAMS-GLOB-ANT.
# Dùng: Rscript tong_hop_huyen.R [thư_mục_tif] [file_ra]
suppressMessages({ library(terra); library(data.table) })
a <- commandArgs(TRUE)
HERE <- "~/NCKH/CNN-SDM_17.11.2026/du_lieu_o_nhiem"
TIF <- if (length(a) >= 1) a[1] else file.path(HERE, "cams_glob_ant/vn")
OUT <- if (length(a) >= 2) a[2] else file.path(HERE, "phat_thai_huyen_G3_dai.csv")
h <- vect(path.expand(file.path(HERE, "ghep_huyen/huyen_G3_638.gpkg")))
h <- h[order(h$g3_row)]
NHOM <- list(tong = "sum", nang_luong = "ene", cong_nghiep = c("ind", "ref", "fef"), giao_thong = "tro",
             phi_duong_bo = "tnr", dan_dung = c("res", "com"), nong_nghiep = c("agl", "ags", "awb"),
             chat_thai = "swd", hang_hai = "shp")
fs <- list.files(path.expand(TIF), "\\.tif$", full.names = TRUE)
res <- rbindlist(lapply(fs, function(f) {
  b <- sub("\\.tif$", "", basename(f)); y <- as.integer(sub(".*_", "", b)); v <- sub("_[0-9]{4}$", "", b)
  r <- rast(f); sec <- sub("_[0-9]{2}$", "", names(r)); mo <- as.integer(sub(".*_", "", names(r)))
  ngay <- c(31, ifelse((y %% 4 == 0 & y %% 100 != 0) | y %% 400 == 0, 29, 28), 31, 30, 31, 30, 31, 31, 30, 31, 30, 31)
  # kg m-2 s-1 -> tấn km-2 năm-1: tổng_tháng(flux * ngày * 86400) * 1e6 / 1e3
  rbindlist(lapply(names(NHOM), function(k) {
    idx <- which(sec %in% NHOM[[k]]); if (!length(idx)) return(NULL)
    ann <- sum(r[[idx]] * ngay[mo[idx]] * 86400 * 1e3)                  # tấn/km²/năm, mỗi ô
    e <- extract(ann, h, fun = mean, exact = TRUE, ID = FALSE)[, 1]
    data.table(g3_row = h$g3_row, name = h$name, province = h$province, bien = v, nam = y, nganh = k, t_km2_nam = e)
  }))
}))
fwrite(res, OUT)
cat("đã ghi", OUT, nrow(res), "dòng;", length(fs), "file; NA:", sum(is.na(res$t_km2_nam)), "\n")
