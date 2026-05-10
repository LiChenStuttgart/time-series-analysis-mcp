suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(deseats))

`%||%` <- function(lhs, rhs) {
  if (is.null(lhs)) rhs else lhs
}

get_opt <- function(opts, name, default = NULL) {
  value <- opts[[name]]
  if (is.null(value)) default else value
}

numeric_vector <- function(value, default = NULL) {
  if (is.null(value)) {
    return(default)
  }
  as.numeric(unlist(value, use.names = FALSE))
}

integer_or_null <- function(value) {
  if (is.null(value)) {
    return(NULL)
  }
  as.integer(value)
}

capture_warnings <- function(expr) {
  warnings <- character()
  value <- withCallingHandlers(
    expr,
    warning = function(w) {
      warnings <<- c(warnings, conditionMessage(w))
      invokeRestart("muffleWarning")
    }
  )
  list(value = value, warnings = warnings)
}

make_smoothing_options <- function(opts, seasonal_period) {
  bwidth <- get_opt(opts, "bwidth")
  deseats::set_options(
    order_poly = as.integer(get_opt(opts, "order_poly", 3)),
    season = as.numeric(seasonal_period),
    kernel_fun = as.character(get_opt(opts, "kernel_fun", "epanechnikov")),
    bwidth = if (is.null(bwidth)) NA_real_ else as.numeric(bwidth),
    boundary_method = as.character(get_opt(opts, "boundary_method", "shorten"))
  )
}

decomposition_args <- function(y, smoothing_options, opts) {
  args <- list(
    y = y,
    smoothing_options = smoothing_options,
    correction_factor = isTRUE(get_opt(opts, "correction_factor", FALSE)),
    autocor = isTRUE(get_opt(opts, "autocor", TRUE)),
    error_model = as.character(get_opt(opts, "error_model", "free")),
    nar_lim = numeric_vector(get_opt(opts, "nar_lim"), c(0, 3)),
    nma_lim = numeric_vector(get_opt(opts, "nma_lim"), c(0, 3)),
    arma_mean = isTRUE(get_opt(opts, "arma_mean", FALSE)),
    inf_criterion = as.character(get_opt(opts, "inf_criterion", "bic"))
  )
  if (!is.null(opts[["bwidth_start"]])) {
    args$bwidth_start <- as.numeric(opts[["bwidth_start"]])
  }
  if (!is.null(opts[["inflation_rate"]])) {
    args$inflation_rate <- as.character(opts[["inflation_rate"]])
  }
  if (!is.null(opts[["drop"]])) {
    args$drop <- as.numeric(opts[["drop"]])
  }
  args
}

forecast_args <- function(y, smoothing_options, opts) {
  ar_order <- integer_or_null(get_opt(opts, "ar_order"))
  ma_order <- integer_or_null(get_opt(opts, "ma_order"))
  args <- list(
    yt = y,
    smoothing_options = smoothing_options,
    arma_options = list(ar_order = ar_order, ma_order = ma_order),
    correction_factor = isTRUE(get_opt(opts, "correction_factor", FALSE)),
    error_model = as.character(get_opt(opts, "error_model", "free")),
    nar_lim = numeric_vector(get_opt(opts, "nar_lim"), c(0, 3)),
    nma_lim = numeric_vector(get_opt(opts, "nma_lim"), c(0, 3)),
    arma_mean = isTRUE(get_opt(opts, "arma_mean", FALSE)),
    inf_criterion = as.character(get_opt(opts, "inf_criterion", "bic"))
  )
  if (!is.null(opts[["bwidth_start"]])) {
    args$bwidth_start <- as.numeric(opts[["bwidth_start"]])
  }
  if (!is.null(opts[["inflation_rate"]])) {
    args$inflation_rate <- as.character(opts[["inflation_rate"]])
  }
  if (!is.null(opts[["drop"]])) {
    args$drop <- as.numeric(opts[["drop"]])
  }
  args
}

make_interval_bands <- function(fc, confidence_levels) {
  interv <- as.matrix(fc@interv)
  if (length(interv) == 1 && is.na(interv[[1]])) {
    return(list())
  }
  col_probs <- suppressWarnings(as.numeric(sub("%", "", colnames(interv), fixed = TRUE)) / 100)
  bands <- list()
  for (confidence in confidence_levels) {
    lower_prob <- (1 - confidence) / 2
    upper_prob <- 1 - lower_prob
    lower_idx <- which.min(abs(col_probs - lower_prob))
    upper_idx <- which.min(abs(col_probs - upper_prob))
    bands[[length(bands) + 1]] <- list(
      confidence_level = confidence,
      lower = as.numeric(interv[, lower_idx]),
      upper = as.numeric(interv[, upper_idx])
    )
  }
  bands
}

process_decompose <- function(series, opts) {
  tryCatch({
    seasonal_period <- as.integer(get_opt(opts, "seasonal_period", 1))
    y <- stats::ts(as.numeric(unlist(series$values, use.names = FALSE)), frequency = seasonal_period)
    smoothing_options <- make_smoothing_options(opts, seasonal_period)
    captured <- capture_warnings({
      do.call(deseats::deseats, decomposition_args(y, smoothing_options, opts))
    })
    est <- captured$value
    list(
      ok = TRUE,
      series_id = series$series_id,
      n_obs = length(y),
      frequency = as.integer(est@frequency),
      observations = as.numeric(est@decomp[, "Observations"]),
      trend = as.numeric(deseats::trend(est)),
      seasonality = as.numeric(deseats::season(est)),
      fitted = as.numeric(fitted(est)),
      residuals = as.numeric(residuals(est)),
      deseasonalized = as.numeric(deseats::deseasonalize(est)),
      detrended = as.numeric(deseats::detrend(est)),
      selected_bandwidth = as.numeric(est@bwidth),
      sum_autocov = as.numeric(est@sum_autocov),
      warnings = as.list(captured$warnings)
    )
  }, error = function(e) {
    list(ok = FALSE, series_id = series$series_id, error = conditionMessage(e))
  })
}

process_forecast <- function(series, opts) {
  tryCatch({
    seasonal_period <- as.integer(get_opt(opts, "seasonal_period", 1))
    y <- stats::ts(as.numeric(unlist(series$values, use.names = FALSE)), frequency = seasonal_period)
    smoothing_options <- make_smoothing_options(opts, seasonal_period)
    seed <- get_opt(opts, "seed")
    if (!is.null(seed)) {
      set.seed(as.integer(seed))
    }
    boot_cores <- suppressWarnings(as.integer(Sys.getenv("TS_MCP_DESEATS_BOOT_CORES", "1")))
    if (is.na(boot_cores) || boot_cores < 1) {
      boot_cores <- 1
    }
    old_cores <- options(mc.cores = boot_cores)
    on.exit(options(old_cores), add = TRUE)

    captured_model <- capture_warnings({
      do.call(deseats::s_semiarma, forecast_args(y, smoothing_options, opts))
    })
    model <- captured_model$value
    confidence_levels <- numeric_vector(get_opt(opts, "confidence_levels"), c(0.95))
    captured_fc <- capture_warnings({
      predict(
        model,
        n.ahead = as.integer(get_opt(opts, "forecast_steps", 1)),
        intervals = TRUE,
        alpha = confidence_levels,
        method = as.character(get_opt(opts, "interval_method", "norm")),
        bootMethod = as.character(get_opt(opts, "bootstrap_method", "simple")),
        npaths = as.integer(get_opt(opts, "npaths", 1000)),
        quant.type = as.integer(get_opt(opts, "quant_type", 8)),
        expo = isTRUE(get_opt(opts, "expo", FALSE)),
        adjust.bias = isTRUE(get_opt(opts, "adjust_bias", TRUE))
      )
    })
    fc <- captured_fc$value
    tail_n <- min(length(y), seasonal_period, 12)
    arma_info <- model@par_model$arma
    list(
      ok = TRUE,
      series_id = series$series_id,
      n_obs = length(y),
      frequency = as.integer(model@nonpar_model@frequency),
      forecast = as.numeric(fc@pred),
      intervals = make_interval_bands(fc, confidence_levels),
      selected_bandwidth = as.numeric(model@nonpar_model@bwidth),
      arma_order = list(as.integer(arma_info[[1]]), 0L, as.integer(arma_info[[2]])),
      decomposition_tail = list(
        trend = as.numeric(utils::tail(deseats::trend(model), tail_n)),
        seasonality = as.numeric(utils::tail(deseats::season(model), tail_n)),
        residuals = as.numeric(utils::tail(residuals(model), tail_n))
      ),
      warnings = as.list(c(captured_model$warnings, captured_fc$warnings))
    )
  }, error = function(e) {
    list(ok = FALSE, series_id = series$series_id, error = conditionMessage(e))
  })
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
  stop("JSON input path is required")
}

payload <- jsonlite::fromJSON(args[[1]], simplifyVector = FALSE)
operation <- payload$operation %||% ""
opts <- payload$options %||% list()
series_list <- payload$series %||% list()

if (operation == "decompose") {
  results <- lapply(series_list, process_decompose, opts = opts)
} else if (operation == "forecast") {
  results <- lapply(series_list, process_forecast, opts = opts)
} else {
  stop(paste0("unsupported operation: ", operation))
}

output <- list(ok = TRUE, operation = operation, results = results)
cat(jsonlite::toJSON(output, auto_unbox = TRUE, null = "null", na = "null", digits = 16))
