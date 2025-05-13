# Make per drug training set
rm(list = ls())
library(data.table)
library(magrittr)

# Read file ####
cpm <- fread(
    file = "datasets/beataml/real/Gene_Counts_CPM.csv",
    check.names = TRUE
)
cpm_mat <- as.matrix(cpm[, -c(1, 2)])
rownames(cpm_mat) <- cpm$Gene

# Transpose ####
cpm_mat <- t(cpm_mat)

## Read and organize the Drug Response data
drug_response <- fread(file = "datasets/beataml/real/Drug_Responses.csv")
drug_response$inhibitor <- make.names(drug_response$inhibitor)
drug_response$lab_id <- make.names(drug_response$lab_id)
colnames(drug_response)[2] <- "labid"

# Drugs in clinical practice
drugs_in_clinical_practice <- c(
    "Gilteritinib..ASP.2215.",
    "Lenalidomide",
    "Midostaurin",
    "Venetoclax"
)

# Create training data for each drug
drug_j <- drugs_in_clinical_practice[1]
for (drug_j in drugs_in_clinical_practice) {
    drug_response_j <- drug_response[inhibitor == drug_j, c("labid", "auc")]

    # Convert the problem to a classification.
    # For educational purposes, we use the median AUC value to define drug
    # response classes.
    drug_response_j[
        , auc_binary := ifelse(test = auc <= median(drug_response_j[, auc]),
            yes = 1,
            no = 0
        )
    ]

    # Keep the samples of the drug
    cpm_matj <- cpm_mat[rownames(cpm_mat) %in% drug_response_j$labid, ]
    # Keep the 50 most variable genes for each drug
    sds_genes <- apply(cpm_matj, MARGIN = 2, sd) %>% sort(decreasing = TRUE)
    genes2keep <- sds_genes[1:50] %>% names()
    cpm_matj <- cpm_matj[, colnames(cpm_mat) %in% sort(genes2keep)]
    
    # Convert to data.table
    cpm_preprocess <- data.table(
        cpm_matj, 
        keep.rownames = TRUE
        )
    colnames(cpm_preprocess)[1] <- "labid"

    # Merge data
    data_all <- merge.data.table(
        x = drug_response_j[, c("labid", "auc_binary")],
        y = cpm_preprocess,
        by = "labid"
    )

    # Write data
    fwrite(
        x = data_all,
        file = paste0("datasets/beataml/real/", drug_j, "_training_data.csv"),
        row.names = FALSE
    )
}
