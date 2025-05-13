# Tutorial Description

In this 3-hour hands-on tutorial, we will explore the application of machine learning techniques for drug response prediction using R. The session is designed for Master’s and PhD students in bioinformatics who want to deepen their understanding of ML workflows and tools in R. We will use the powerful `caret` package for building, training, evaluating and interpreting classification models. Additionally, we will introduce the `h2o` package for scalable modeling.

Participants will work with the [BeatAML](https://www.nature.com/articles/s41586-018-0623-z#Sec38) dataset published in Nature 2018, and available in the supplementary material of the paper. BeatAML is a rich pharmacogenomic resource derived from primary samples of acute myeloid leukemia (AML) patients.

This dataset includes:

-   RNA-seq gene expression profiles (RPKM, CPM)

-   Drug response measurements (e.g. IC50, AUC)

-   Clinical annotations

-   Variants

By the end of the tutorial, participants will:

-   Understand data pre-processing steps for transcriptomics data

-   Train multiple ML models (Elastic Net, KNN, Random Forest, Gradient Boosting Machines, Support Vector Machines) using `caret`. More information about `caret` can be found here: <https://topepo.github.io/caret/index.html>

-   Evaluate model performance using cross-validation and test sets and several metrics (Confusion matrices, Accuracy, Area Under the ROC curve, etc)

-   Identify important features

-   Explore deep learning and AutoML using `h2o`. More information about `h2o` can be found here: <https://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html>

At the end of the tutorial, participants will be given a **mini-competition assignment** to apply what they've learned. The team with the best-performing solution will have the opportunity to present their results during the closing session of the conference.

## Software and data requirements

To fully participate in this hands-on tutorial, please ensure the following software and packages are installed before the session begins:

1.  R and RStudio (latest versions)

2.  Required R Packages You can install all required packages using the command below:

```{r}
packages <- c("data.table", "caret", "ggplot2", "pROC", "doParallel", 
              "magrittr", "h2o", "e1071", "randomForest", "gbm", 
              "glmnet", "kernlab")
install.packages(packages)
```

3.  System Requirements

    A machine with at least 8 GB RAM

    Multicore CPU recommended for parallel processing

4.  Dataset

    Please download the BeatAML dataset - "Supplementary Tables" - (from the supplementary materials of the [original publication](https://www.nature.com/articles/s41586-018-0623-z#Sec38)).
