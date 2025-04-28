install.packages("h2o")

if (!require("BiocManager", quietly = TRUE))
    install.packages("BiocManager")

BiocManager::install("curatedTCGAData")
BiocManager::install("SummarizedExperiment")
BiocManager::install("TCGAutils")


