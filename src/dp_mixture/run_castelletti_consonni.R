#!/usr/bin/env Rscript

## Run the unmodified Castelletti--Consonni MCMC code on one saved seed.

get_script_dir <- function() {
  full_args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", full_args, value = TRUE)
  if (length(file_arg) != 1L) {
    return(normalizePath(".", mustWork = TRUE))
  }
  dirname(normalizePath(sub("^--file=", "", file_arg), mustWork = TRUE))
}


usage <- function() {
  cat(paste(
    "Usage:",
    "  Rscript run_castelletti_consonni.R \\",
    "    --data-dir <seed_directory> \\",
    "    --out-dir <output_directory> [options]",
    "",
    "Options:",
    "  --code-dir <directory>  Author code directory",
    "  --S <integer>           Total MCMC iterations (default: 2500)",
    "  --burn <integer>        Burn-in iterations (default: 500)",
    "  --a-alpha <number>      Gamma shape for DP concentration (default: 1)",
    "  --b-alpha <number>      Gamma rate for DP concentration (default: 3)",
    "  --a-pi <number>         Beta shape for edge probability (default: 1)",
    "  --b-pi <number>         Beta shape; default is 2*(q-2)/3",
    "  --y-set <csv>           One-based response nodes (default: 1)",
    "  --seed <integer>        Dataset seed recorded in metadata",
    "  --help                  Show this message",
    sep = "\n"
  ))
}


parse_cli <- function(args, script_dir) {
  options <- list(
    data_dir = NULL,
    out_dir = NULL,
    code_dir = file.path(
      script_dir, "castelletti_consonni_bnp_mixture_causal_dags"
    ),
    S = 2500L,
    burn = 500L,
    a_alpha = 1,
    b_alpha = 3,
    a_pi = 1,
    b_pi = NA_real_,
    y_set = 1L,
    seed = NA_integer_
  )

  if ("--help" %in% args) {
    usage()
    quit(status = 0L)
  }

  key_map <- c(
    "--data-dir" = "data_dir",
    "--out-dir" = "out_dir",
    "--code-dir" = "code_dir",
    "--S" = "S",
    "--burn" = "burn",
    "--a-alpha" = "a_alpha",
    "--b-alpha" = "b_alpha",
    "--a-pi" = "a_pi",
    "--b-pi" = "b_pi",
    "--y-set" = "y_set",
    "--seed" = "seed"
  )

  index <- 1L
  while (index <= length(args)) {
    key <- args[[index]]
    if (!key %in% names(key_map)) {
      stop("Unknown argument: ", key, call. = FALSE)
    }
    if (index == length(args)) {
      stop("Missing value after ", key, call. = FALSE)
    }
    options[[key_map[[key]]]] <- args[[index + 1L]]
    index <- index + 2L
  }

  if (is.null(options$data_dir) || is.null(options$out_dir)) {
    usage()
    stop("--data-dir and --out-dir are required.", call. = FALSE)
  }

  options$S <- as.integer(options$S)
  options$burn <- as.integer(options$burn)
  options$a_alpha <- as.numeric(options$a_alpha)
  options$b_alpha <- as.numeric(options$b_alpha)
  options$a_pi <- as.numeric(options$a_pi)
  options$b_pi <- as.numeric(options$b_pi)
  options$seed <- as.integer(options$seed)
  options$y_set <- as.integer(
    strsplit(as.character(options$y_set), ",", fixed = TRUE)[[1L]]
  )
  options
}


validate_scalar <- function(value, name, lower = -Inf, strict = FALSE) {
  if (length(value) != 1L || is.na(value) || !is.finite(value)) {
    stop(name, " must be one finite number.", call. = FALSE)
  }
  invalid <- if (strict) value <= lower else value < lower
  if (invalid) {
    comparison <- if (strict) ">" else ">="
    stop(name, " must be ", comparison, " ", lower, ".", call. = FALSE)
  }
}


is_acyclic_adjacency <- function(adjacency) {
  adjacency <- adjacency != 0
  diag(adjacency) <- FALSE
  indegree <- colSums(adjacency)
  available <- which(indegree == 0)
  visited <- 0L

  while (length(available) > 0L) {
    node <- available[[1L]]
    available <- available[-1L]
    visited <- visited + 1L
    children <- which(adjacency[node, ])
    for (child in children) {
      indegree[[child]] <- indegree[[child]] - 1L
      if (indegree[[child]] == 0L) {
        available <- c(available, child)
      }
    }
  }
  visited == nrow(adjacency)
}


authors_similarity_partition <- function(simil_probs) {
  ## This reproduces the point-estimation rule in example_for_mcmc.R.
  n <- nrow(simil_probs)
  simil_mat <- round(simil_probs)
  cluster_ids <- rep(NA_integer_, n)
  for (i in n:1L) {
    cluster_ids[simil_mat[i, ] == 1] <- i
  }
  if (anyNA(cluster_ids)) {
    stop("Author similarity rule left at least one subject unassigned.")
  }
  as.integer(factor(cluster_ids))
}


prepare_author_workspace <- function(code_dir) {
  source_files <- c(
    "mcmc_mixture_dags.R",
    "sample_normal_dag_wishart.R",
    "norm_constant_normal_dag_wishart.R",
    "move_dag.R",
    "sample_from_baseline.R"
  )
  missing_files <- source_files[!file.exists(file.path(code_dir, source_files))]
  if (length(missing_files) > 0L) {
    stop(
      "Missing author source files: ", paste(missing_files, collapse = ", "),
      call. = FALSE
    )
  }

  workspace <- tempfile("castelletti_consonni_")
  dir.create(workspace)
  copied_names <- c(
    "mcmc_mixture_dags.R",
    "sample_normal_dag_wishart.r",
    "norm_constant_normal_dag_wishart.r",
    "move_dag.r",
    "sample_from_baseline.r"
  )
  copied <- file.copy(
    file.path(code_dir, source_files),
    file.path(workspace, copied_names),
    overwrite = FALSE
  )
  if (!all(copied)) {
    stop("Failed to create the temporary author-code workspace.", call. = FALSE)
  }
  workspace
}


write_run_metadata <- function(path, options, input_path, n, q, b_pi, timing) {
  lines <- c(
    paste0("dataset_seed: ", ifelse(is.na(options$seed), "NA", options$seed)),
    paste0("input_file: ", input_path),
    paste0("n_subjects: ", n),
    paste0("n_variables: ", q),
    paste0("S: ", options$S),
    paste0("burn: ", options$burn),
    paste0("a_alpha: ", options$a_alpha),
    paste0("b_alpha: ", options$b_alpha),
    paste0("a_pi: ", options$a_pi),
    paste0("b_pi: ", b_pi),
    paste0("y_set: ", paste(options$y_set, collapse = ",")),
    "edge_probability_threshold: 0.5",
    "cluster_extraction: author_round_similarity_rule",
    "mcmc_initialization_seed_internal_to_author_code: 123",
    paste0("elapsed_seconds: ", unname(timing[["elapsed"]])),
    paste0("R_version: ", R.version.string),
    paste0("completed_utc: ", format(Sys.time(), tz = "UTC", usetz = TRUE))
  )
  writeLines(lines, path, useBytes = TRUE)
}


main <- function() {
  script_dir <- get_script_dir()
  options <- parse_cli(commandArgs(trailingOnly = TRUE), script_dir)

  validate_scalar(options$S, "S", lower = 2)
  validate_scalar(options$burn, "burn", lower = 1)
  if (options$burn >= options$S) {
    stop("burn must be smaller than S.", call. = FALSE)
  }
  for (name in c("a_alpha", "b_alpha", "a_pi")) {
    validate_scalar(options[[name]], name, lower = 0, strict = TRUE)
  }

  data_dir <- normalizePath(options$data_dir, mustWork = TRUE)
  code_dir <- normalizePath(options$code_dir, mustWork = TRUE)
  dir.create(options$out_dir, recursive = TRUE, showWarnings = FALSE)
  out_dir <- normalizePath(options$out_dir, mustWork = TRUE)
  input_path <- file.path(data_dir, "castelletti_input.csv")
  labels_path <- file.path(data_dir, "cluster_labels.csv")

  if (!file.exists(input_path) || !file.exists(labels_path)) {
    stop(
      "Expected castelletti_input.csv and cluster_labels.csv in ", data_dir,
      call. = FALSE
    )
  }

  X <- as.matrix(read.csv(input_path, header = FALSE, check.names = FALSE))
  storage.mode(X) <- "double"
  true_labels <- read.csv(labels_path, header = TRUE, check.names = FALSE)
  if (nrow(X) != nrow(true_labels)) {
    stop("Input rows and cluster-label rows do not match.", call. = FALSE)
  }
  if (any(!is.finite(X))) {
    stop("The input matrix contains non-finite values.", call. = FALSE)
  }

  n <- nrow(X)
  q <- ncol(X)
  if (q < 2L) {
    stop("The Castelletti--Consonni implementation requires q >= 2.", call. = FALSE)
  }

  ## The author's pa() helper refers to q in .GlobalEnv rather than deriving it
  ## from its adjacency-matrix argument. Their example scripts therefore define
  ## q globally before starting MCMC. Mirror that interface requirement here
  ## without modifying the downloaded author source files.
  assign("q", as.integer(q), envir = .GlobalEnv)

  if (any(options$y_set < 1L | options$y_set > q)) {
    stop("Every y-set node must lie in {1, ..., q}.", call. = FALSE)
  }
  b_pi <- if (is.na(options$b_pi)) 2 * (q - 2) / 3 else options$b_pi
  validate_scalar(b_pi, "b_pi", lower = 0, strict = TRUE)

  required_packages <- c("pcalg", "gRbase", "mvtnorm", "abind")
  available <- vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
  if (!all(available)) {
    stop(
      "Missing required R packages: ",
      paste(required_packages[!available], collapse = ", "),
      call. = FALSE
    )
  }

  author_workspace <- prepare_author_workspace(code_dir)
  on.exit(unlink(author_workspace, recursive = TRUE, force = TRUE), add = TRUE)
  original_working_dir <- getwd()
  on.exit(setwd(original_working_dir), add = TRUE)
  setwd(author_workspace)
  source("mcmc_mixture_dags.R")

  cat(
    sprintf(
      "Running Castelletti--Consonni: n=%d, q=%d, S=%d, burn=%d, b_pi=%.8g\n",
      n, q, options$S, options$burn, b_pi
    )
  )
  timing <- system.time({
    output <- mcmc_mixture_dags(
      X = X,
      S = options$S,
      burn = options$burn,
      a_alpha = options$a_alpha,
      b_alpha = options$b_alpha,
      a_pi = options$a_pi,
      b_pi = b_pi,
      y_set = options$y_set
    )
  })

  estimated_labels <- authors_similarity_partition(output$simil_probs)
  estimated_dags <- 1L * (output$graph_probs > 0.5)
  for (i in seq_len(n)) {
    diag(estimated_dags[, , i]) <- 0L
  }
  dag_is_acyclic <- vapply(
    seq_len(n),
    function(i) is_acyclic_adjacency(estimated_dags[, , i]),
    logical(1)
  )
  edge_counts <- vapply(
    seq_len(n), function(i) sum(estimated_dags[, , i]), integer(1)
  )

  saveRDS(output, file.path(out_dir, "mcmc_output.rds"))
  saveRDS(output$graph_probs, file.path(out_dir, "graph_probabilities.rds"))
  saveRDS(estimated_dags, file.path(out_dir, "estimated_dags_threshold_0.5.rds"))
  write.csv(
    output$simil_probs,
    file.path(out_dir, "posterior_similarity.csv"),
    row.names = FALSE
  )

  assignments <- data.frame(
    subject_id = true_labels$subject_id,
    true_cluster_label = true_labels$cluster_label,
    estimated_cluster_label = estimated_labels,
    estimated_edge_count = edge_counts,
    estimated_graph_is_acyclic = dag_is_acyclic
  )
  write.csv(
    assignments,
    file.path(out_dir, "cluster_assignments.csv"),
    row.names = FALSE
  )

  edge_table <- expand.grid(
    from_node = seq_len(q),
    to_node = seq_len(q),
    subject_id = seq_len(n)
  )
  edge_table$posterior_probability <- as.vector(output$graph_probs)
  edge_table$selected_at_0.5 <- as.vector(estimated_dags)
  write.csv(
    edge_table,
    file.path(out_dir, "subject_edge_probabilities.csv"),
    row.names = FALSE
  )

  for (response_name in names(output$est_causal)) {
    response_node <- sub("response y = ", "", response_name, fixed = TRUE)
    causal_table <- data.frame(
      subject_id = seq_len(n),
      output$est_causal[[response_name]],
      check.names = FALSE
    )
    names(causal_table)[-1L] <- paste0("node_", seq_len(q))
    write.csv(
      causal_table,
      file.path(out_dir, paste0("causal_effects_response_", response_node, ".csv")),
      row.names = FALSE
    )
  }

  runtime <- data.frame(
    user_seconds = unname(timing[["user.self"]]),
    system_seconds = unname(timing[["sys.self"]]),
    elapsed_seconds = unname(timing[["elapsed"]])
  )
  write.csv(runtime, file.path(out_dir, "runtime.csv"), row.names = FALSE)
  write_run_metadata(
    file.path(out_dir, "run_metadata.txt"),
    options, input_path, n, q, b_pi, timing
  )

  cat(
    sprintf(
      "Completed in %.2f seconds; estimated %d clusters; %d/%d thresholded graphs are acyclic.\n",
      unname(timing[["elapsed"]]), length(unique(estimated_labels)),
      sum(dag_is_acyclic), n
    )
  )
  cat("Outputs written to: ", out_dir, "\n", sep = "")
}


main()
