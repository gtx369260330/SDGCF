#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(grid)
})

out_dir <- file.path("results", "figures", "sdgcf_overall_structure")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

width_in <- 11.2
height_in <- 6.6

pal <- list(
  ink = "#172033",
  muted = "#617083",
  faint = "#F6F8FB",
  line = "#C8D2DE",
  blue = "#2F6F9F",
  blue_fill = "#EAF4FB",
  teal = "#178C8C",
  teal_fill = "#EAF7F6",
  gold = "#C9932F",
  gold_fill = "#FFF6DF",
  red = "#B95149",
  red_fill = "#FFF0ED",
  green = "#4D8B58",
  green_fill = "#EEF8F0",
  purple = "#6A5AA8",
  purple_fill = "#F3F0FA"
)

u <- function(x) unit(x, "native")

draw_text <- function(label, x, y, size = 9, col = pal$ink, face = "plain",
                      just = "center", lineheight = 0.92) {
  grid.text(
    label, x = u(x), y = u(y), just = just,
    gp = gpar(fontsize = size, col = col, fontface = face, lineheight = lineheight)
  )
}

draw_round_box <- function(x, y, w, h, label, subtitle = NULL, fill = "white",
                           border = pal$line, lwd = 1.2, label_size = 8.4,
                           subtitle_size = 7.1, title_face = "bold") {
  grid.roundrect(
    x = u(x), y = u(y), width = u(w), height = u(h),
    r = unit(0.08, "snpc"),
    gp = gpar(fill = fill, col = border, lwd = lwd)
  )
  if (is.null(subtitle)) {
    draw_text(label, x, y, label_size, pal$ink, title_face)
  } else {
    draw_text(label, x, y + h * 0.17, label_size, pal$ink, title_face)
    draw_text(subtitle, x, y - h * 0.16, subtitle_size, pal$muted, "plain")
  }
}

draw_section <- function(x, y, w, h, title, accent) {
  grid.roundrect(
    x = u(x), y = u(y), width = u(w), height = u(h),
    r = unit(0.075, "snpc"),
    gp = gpar(fill = pal$faint, col = "#E1E6EE", lwd = 1.0)
  )
  grid.roundrect(
    x = u(x - w / 2 + 1.4), y = u(y), width = u(0.9), height = u(h - 4),
    r = unit(0.04, "snpc"), gp = gpar(fill = accent, col = NA)
  )
  draw_text(title, x - w / 2 + 4.2, y + h / 2 - 4.1, 8.2, accent, "bold", just = "left")
}

draw_arrow <- function(x1, y1, x2, y2, col = pal$muted, lwd = 1.1,
                       curvature = 0, arrow_len = 0.10) {
  if (abs(curvature) < 1e-6) {
    grid.lines(
      x = u(c(x1, x2)), y = u(c(y1, y2)),
      arrow = arrow(type = "closed", length = unit(arrow_len, "inches")),
      gp = gpar(col = col, lwd = lwd, lineend = "round")
    )
  } else {
    grid.curve(
      x1 = u(x1), y1 = u(y1), x2 = u(x2), y2 = u(y2),
      curvature = curvature,
      arrow = arrow(type = "closed", length = unit(arrow_len, "inches")),
      gp = gpar(col = col, lwd = lwd, lineend = "round")
    )
  }
}

draw_waveform <- function(cx, cy, w, h, col) {
  xs <- seq(cx - w / 2 + 1.6, cx + w / 2 - 1.8, length.out = 90)
  phase <- if (cy > 51) 0 else if (cy > 42) 1.1 else 2.2
  ys <- cy + 0.72 * sin(seq(0, 5.8 * pi, length.out = 90) + phase) +
    0.25 * sin(seq(0, 18 * pi, length.out = 90) + phase)
  grid.lines(x = u(xs), y = u(ys), gp = gpar(col = col, lwd = 1.0))
}

draw_node <- function(x, y, label, fill = pal$teal_fill, border = pal$teal) {
  grid.circle(x = u(x), y = u(y), r = u(3.2), gp = gpar(fill = fill, col = border, lwd = 1.3))
  draw_text(label, x, y, 7.2, pal$ink, "bold")
}

draw_heatmap <- function(x, y, cell = 2.15) {
  values <- matrix(c(
    0.88, 0.56, 0.66,
    0.48, 0.83, 0.54,
    0.61, 0.46, 0.86
  ), nrow = 3, byrow = TRUE)
  cols <- colorRampPalette(c("#EDF6F9", "#7DB6C7", "#1E6F8F"))(100)
  for (i in 1:3) {
    for (j in 1:3) {
      cx <- x + (j - 2) * cell
      cy <- y + (2 - i) * cell
      grid.rect(
        x = u(cx), y = u(cy), width = u(cell * 0.88), height = u(cell * 0.88),
        gp = gpar(fill = cols[max(1, min(100, round(values[i, j] * 100)))], col = "white", lwd = 0.6)
      )
    }
  }
  grid.rect(x = u(x), y = u(y), width = u(cell * 3.02), height = u(cell * 3.02),
            gp = gpar(fill = NA, col = pal$teal, lwd = 0.9))
  draw_text("sample-wise\nA_mn", x, y - cell * 2.35, 6.5, pal$muted)
}

draw_pill <- function(x, y, label, fill, border, w = 6.2) {
  grid.roundrect(
    x = u(x), y = u(y), width = u(w), height = u(4.1),
    r = unit(0.05, "snpc"), gp = gpar(fill = fill, col = border, lwd = 1.0)
  )
  draw_text(label, x, y, 7.8, pal$ink, "bold")
}

draw_poly_arrow <- function(xs, ys, col = pal$muted, lwd = 1.0, arrow_len = 0.10) {
  grid.lines(
    x = u(xs), y = u(ys),
    arrow = arrow(type = "closed", length = unit(arrow_len, "inches")),
    gp = gpar(col = col, lwd = lwd, lineend = "round", linejoin = "round")
  )
}

draw_tile <- function(x, y, label, fill, border) {
  grid.roundrect(
    x = u(x), y = u(y), width = u(19), height = u(7.5),
    r = unit(0.055, "snpc"), gp = gpar(fill = fill, col = border, lwd = 1.0)
  )
  draw_text(label, x, y, 7.3, pal$ink, "plain")
}

draw_figure <- function() {
  grid.newpage()
  pushViewport(viewport(xscale = c(0, 100), yscale = c(0, 100)))

  grid.rect(gp = gpar(fill = "white", col = NA))
  draw_text("Sample-wise Dynamic Graph Concatenation Fusion (SDGCF)", 50, 96, 15.5, pal$ink, "bold")
  draw_text("Multimodal EEG-EOG sleep staging: from subject-level epochs to dynamic inter-modality fusion and validation", 50, 91.7, 8.6, pal$muted)

  draw_section(17.5, 57.5, 29.0, 60.5, "Input and epoch construction", pal$blue)
  draw_section(50.0, 57.5, 31.5, 60.5, "SDGCF representation", pal$teal)
  draw_section(82.7, 57.5, 29.0, 60.5, "Prediction and evidence", pal$green)

  draw_round_box(
    17.5, 76.8, 22.8, 9.8,
    "Sleep-EDF Expanded",
    "100 subjects; 237,950 epochs\nEEG Fpz-Cz, EEG Pz-Oz, horizontal EOG",
    pal$blue_fill, pal$blue, label_size = 8.2, subtitle_size = 6.6
  )
  draw_round_box(
    17.5, 64.2, 22.8, 8.6,
    "Preprocessing",
    "30 s epochs; normalization\nsubject-level train/val/test split",
    "white", "#A8BBD0", label_size = 8.0, subtitle_size = 6.7
  )
  draw_arrow(17.5, 71.7, 17.5, 68.6, pal$blue, 1.1)

  channel_y <- c(52.6, 43.7, 34.8)
  channel_names <- c("EEG Fpz-Cz", "EEG Pz-Oz", "Horizontal EOG")
  channel_cols <- c(pal$blue, "#4A82B0", pal$purple)
  for (i in seq_along(channel_y)) {
    draw_round_box(17.5, channel_y[i], 22.8, 6.7, channel_names[i], NULL, "white", "#B8C6D6", label_size = 7.2)
    draw_waveform(12.0, channel_y[i], 7.5, 4.0, channel_cols[i])
    draw_arrow(29.0, channel_y[i], 34.2, channel_y[i], "#8CA0B7", 0.9)
  }

  encoder_y <- channel_y
  node_y <- c(59.0, 47.0, 35.0)
  node_names <- c("Fpz\nnode", "Pz\nnode", "EOG\nnode")
  for (i in seq_along(encoder_y)) {
    draw_round_box(
      39.2, encoder_y[i], 9.8, 6.7,
      "Multi-scale\nencoder",
      "k = 3 / 7 / 15",
      pal$teal_fill, pal$teal, label_size = 6.8, subtitle_size = 5.7
    )
    draw_arrow(44.3, encoder_y[i], 47.7, node_y[i], pal$teal, 0.9, curvature = ifelse(i == 2, 0, ifelse(i == 1, -0.15, 0.15)))
  }

  draw_node(51.1, node_y[1], node_names[1])
  draw_node(48.1, node_y[2], node_names[2])
  draw_node(54.1, node_y[3], node_names[3])
  grid.curve(x1 = u(51.1), y1 = u(node_y[1] - 3.1), x2 = u(48.1), y2 = u(node_y[2] + 3.1),
             curvature = 0.10, gp = gpar(col = pal$teal, lwd = 1.2))
  grid.curve(x1 = u(51.1), y1 = u(node_y[1] - 3.1), x2 = u(54.1), y2 = u(node_y[3] + 3.1),
             curvature = -0.16, gp = gpar(col = pal$teal, lwd = 1.2))
  grid.curve(x1 = u(48.1 + 3.0), y1 = u(node_y[2] - 0.8), x2 = u(54.1 - 3.0), y2 = u(node_y[3] + 0.8),
             curvature = -0.12, gp = gpar(col = pal$teal, lwd = 1.2))
  draw_text("modality-node graph", 51.1, 66.2, 7.2, pal$muted, "bold")

  draw_round_box(
    61.5, 55.1, 9.7, 14.2,
    "Dynamic\ngraph\nattention",
    NULL, pal$gold_fill, pal$gold, label_size = 7.2
  )
  draw_heatmap(61.5, 43.0, 1.75)
  draw_arrow(56.8, 52.3, 59.0, 52.3, pal$gold, 1.0)
  draw_arrow(61.5, 36.7, 61.5, 33.7, pal$gold, 1.0)

  draw_round_box(
    50.0, 29.2, 25.0, 7.8,
    "Graph-updated modality nodes",
    "residual update with learned inter-channel context",
    "white", "#A4B6C8", label_size = 7.8, subtitle_size = 6.2
  )
  draw_poly_arrow(c(62.8, 66.0, 66.0, 68.6), c(29.2, 29.2, 61.0, 61.0), pal$muted, 1.0)

  draw_round_box(
    75.2, 61.0, 12.5, 9.8,
    "Direct node\nconcatenation",
    NULL, pal$red_fill, pal$red, label_size = 7.3
  )
  draw_round_box(
    90.0, 61.0, 10.5, 9.8,
    "Two-layer\nclassifier",
    NULL, pal$green_fill, pal$green, label_size = 7.4
  )
  draw_arrow(81.8, 61.0, 84.4, 61.0, pal$muted, 1.0)

  draw_text("Sleep stage output", 83.0, 49.7, 8.0, pal$muted, "bold")
  stage_x <- c(72.0, 77.6, 83.2, 88.8, 94.4)
  stage_y <- rep(43.8, 5)
  stage_lab <- c("W", "N1", "N2", "N3", "REM")
  for (i in seq_along(stage_lab)) {
    draw_pill(stage_x[i], stage_y[i], stage_lab[i], "white", "#B9C7D4")
  }
  draw_arrow(90.0, 55.8, 86.4, 46.4, pal$green, 1.0, curvature = 0.08)

  draw_round_box(
    82.7, 27.7, 24.7, 13.6,
    "Model validation",
    "main comparison; ablation\nper-class metrics; ROC/PR\nconfusion, errors, confidence\nmissing/noisy-modality robustness",
    "white", "#AFC2B4", label_size = 8.0, subtitle_size = 6.35
  )
  draw_arrow(86.0, 35.7, 86.0, 40.9, pal$green, 0.9)

  draw_round_box(
    50.0, 13.7, 91.0, 12.7,
    "Training objective and regularization",
    NULL, "#FAFBFC", "#E1E6EE", label_size = 8.6
  )
  draw_tile(19.5, 8.7, "weighted label-smoothed\ncross-entropy", pal$blue_fill, pal$blue)
  draw_tile(39.7, 8.7, "focal loss for\nhard examples", pal$gold_fill, pal$gold)
  draw_tile(59.9, 8.7, "auxiliary modality\nclassification heads", pal$purple_fill, pal$purple)
  draw_tile(80.1, 8.7, "attention entropy +\nnode diversity terms", pal$teal_fill, pal$teal)
  draw_arrow(28.9, 8.7, 30.0, 8.7, pal$muted, 0.7, arrow_len = 0.08)
  draw_arrow(49.1, 8.7, 50.2, 8.7, pal$muted, 0.7, arrow_len = 0.08)
  draw_arrow(69.3, 8.7, 70.4, 8.7, pal$muted, 0.7, arrow_len = 0.08)

  draw_text("Descriptive attention analysis supports interpretation of predictive fusion; it is not treated as causal evidence.", 50, 2.8, 6.5, "#7A8696")

  popViewport()
}

save_device <- function(file, device) {
  device(file)
  draw_figure()
  dev.off()
}

save_device(file.path(out_dir, "sdgcf_overall_structure.svg"), function(file) {
  svg(filename = file, width = width_in, height = height_in, bg = "white")
})

save_device(file.path(out_dir, "sdgcf_overall_structure.pdf"), function(file) {
  pdf(file = file, width = width_in, height = height_in, bg = "white", useDingbats = FALSE)
})

save_device(file.path(out_dir, "sdgcf_overall_structure.png"), function(file) {
  png(filename = file, width = width_in, height = height_in, units = "in", res = 600, bg = "white")
})

save_device(file.path(out_dir, "sdgcf_overall_structure.tiff"), function(file) {
  tiff(filename = file, width = width_in, height = height_in, units = "in", res = 600,
       bg = "white", compression = "lzw")
})

cat(normalizePath(out_dir, winslash = "/", mustWork = TRUE), "\n")
