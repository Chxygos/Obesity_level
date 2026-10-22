# Obesity Pattern Discovery Using Unsupervised Machine Learning

An ML clustering project on the estimation of Obesity Levels based On lifestyle and Physical Conditions.

### K-Means vs. Agglomerative Clustering | Python · Scikit-learn · PCA · t-SNE

**Dataset:** UCI Machine Learning Repository - *Estimation of Obesity Levels Based on Eating Habits and Physical Condition*  
**Tools:** Python, Jupyter Notebook, pandas, NumPy, Matplotlib, Seaborn, scikit-learn  

---

## Project overview

This project investigates whether unsupervised machine learning can reveal meaningful groups of individuals based on their **eating habits, physical characteristics and lifestyle behaviours**. Rather than training a classifier to predict an existing obesity category, I excluded the known obesity-level label from the clustering features and examined the structure discovered by the algorithms.

I implemented and compared **K-Means** and **Agglomerative Hierarchical Clustering**, explored **PCA** and **t-SNE** to visualise the groups, and evaluated clustering structure with the **Silhouette Score**.

> **Headline result:** K-Means achieved a Silhouette Score of **0.149** with **6 clusters**, compared with **0.096** for Agglomerative Clustering with **7 clusters**. K-Means performed better under this measure, although **neither result demonstrates strongly separated groups**.

## Problem and objectives

Obesity-related characteristics can vary across individuals and may not form neat, distinct categories in behavioural data. The objectives were to:

1. Explore a public dataset of physical attributes, dietary habits and lifestyle patterns.
2. Clean and transform mixed numerical and categorical attributes for distance-based clustering.
3. Identify candidate groupings using K-Means and hierarchical clustering.
4. Visualise high-dimensional clusters with PCA and t-SNE.
5. Compare cluster cohesion and separation using the Silhouette Score.
6. Discuss how useful and how limited these groupings may be for exploratory health-data analysis.

**Scope:** This is a clustering and data-exploration project, **not** a clinically validated obesity-risk prediction system or diagnostic tool.

## Dataset

**Source:** [UCI Machine Learning Repository — Estimation of Obesity Levels](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition)

The dataset describes individuals from **Mexico, Peru and Colombia** and contains:

| Characteristic | Reported value |
|---|---|
| Initial observations | **2,111** |
| Initial attributes | **17**, including the obesity-level label |
| Missing values | **None** reported |
| Duplicated rows | **24**, identified and removed during preprocessing |
| Data origin | Approximately **77% synthetic** and **23% collected via a web platform**, according to the dataset documentation quoted in the report |

Feature groups include:

- **Physical and demographic:** gender, age, height, weight, and family history of overweight.
- **Food and hydration:** frequent consumption of high-calorie food (`FAVC`), vegetable consumption (`FCVC`), number of main meals (`NCP`), eating between meals (`CAEC`), and daily water intake (`CH2O`).
- **Lifestyle and activity:** smoking (`SMOKE`), calorie monitoring (`SCC`), physical activity (`FAF`), time using technology (`TUE`), alcohol consumption (`CALC`), and transport mode (`MTRANS`).
- **Reference label (excluded from clustering features):** `NObeyesdad`, the dataset's existing obesity-level category.

The reference label contains categories such as insufficient weight, normal weight, two overweight levels and three obesity types. **The label was removed before clustering** to preserve the unsupervised modelling objective. It was used in parts of the exploratory visualisation, not as a clustering predictor.

> **Data limitation:** Because much of the dataset is synthetic, patterns may not generalise to actual clinical or population samples.


## 1. Exploratory data analysis

I inspected the dataset structure, attribute types, null counts and duplicate rows. The dataset initially contained **2,111 records and 17 columns**; the report identifies **24 duplicate records**, which were removed.

Visual explorations included:

- Relationships among **age, height and weight**.
- **Gender distribution**.
- Height–weight scatter plots coloured by the original obesity-level reference categories, for exploratory context.

These analyses helped assess the spread and overlap of physical and behavioural attributes before applying clustering methods.

## 2. Feature engineering and preprocessing

Distance-based clustering requires a suitable numeric feature representation. The reported workflow therefore included:

- Removing `NObeyesdad` from the modelling features.
- Mapping binary categorical features into numeric values.
- Applying ordinal encoding to ordered frequency-type categories, including `CAEC` and `CALC`.
- One-hot encoding transportation categories (`MTRANS`).
- Applying `StandardScaler` to selected numeric/non-binary attributes, while leaving one-hot encoded indicators unscaled in the reported implementation.

**Modelling consideration:** Encoding choices and the relative weighting of categorical and continuous variables influence Euclidean distances. These choices should be revisited in any follow-up experiment.

## 3. K-Means clustering

I investigated candidate cluster counts using the **Elbow Method**, plotting the within-cluster sum of squares (WCSS) against different values of `k`. The report selected **k = 6** and fitted K-Means with `k-means++` initialisation and `random_state=42`.

**Result:** 6 clusters; **Silhouette Score = 0.149**.

This score suggests weak overall separation: observations within different clusters were often close in feature space.

## 4. Agglomerative hierarchical clustering

I examined hierarchical clustering using a **dendrogram** to assess potential cluster groupings. The report selected **7 clusters** and fitted an Agglomerative Clustering model using **Euclidean distance** and **Ward linkage**.

**Result:** 7 clusters; **Silhouette Score = 0.096**.

Under the reported setup, this method produced weaker cluster cohesion and separation than K-Means.

## 5. Visualising high-dimensional clusters

### Principal Component Analysis (PCA)

I projected the feature space onto two principal components for cluster visualisation. The two-dimensional representation explained **approximately 33% of the variance**, and several groups appeared to overlap.

**Interpretation:** Two-dimensional PCA plots provide a useful summary but cannot fully represent the separation of a higher-dimensional dataset.

### t-Distributed Stochastic Neighbor Embedding (t-SNE)

I also applied **t-SNE** to explore local structure and create more visually interpretable scatter plots of the cluster assignments.

**Interpretation:** t-SNE can reveal local patterns, but apparent gaps in a t-SNE plot should not be treated as independent proof that clusters are well separated. The Silhouette Score was the reported quantitative comparison measure.

## Results and model comparison

| Evaluation criterion | K-Means | Agglomerative clustering |
|---|---:|---:|
| Chosen number of clusters | **6** | **7** |
| Cluster-count selection | Elbow Method | Dendrogram |
| Silhouette Score | **0.149** | **0.096** |
| Relative outcome | Higher score | Lower score |

**Key findings**

- **K-Means outperformed Agglomerative Clustering** on the reported Silhouette Score.
- **Both algorithms showed weak clustering structure**, so the observed groups should be treated as exploratory rather than naturally distinct obesity categories.
- Two-component PCA retained only about **one-third of the variance**, making dimensionality reduction a significant consideration when interpreting plots.
- The results highlight the challenges of discovering discrete groups in mixed lifestyle and physical-characteristic data.


## Skills demonstrated

| Area | Techniques and tools |
|---|---|
| Data preparation | pandas, duplicate detection, feature selection |
| Exploratory analysis | Matplotlib, Seaborn, descriptive visualisation |
| Feature engineering | Binary mapping, OrdinalEncoder, one-hot encoding |
| Scaling | StandardScaler |
| Unsupervised learning | K-Means, Agglomerative Clustering |
| Hyperparameter selection | Elbow Method, dendrogram interpretation |
| Dimensionality reduction | PCA, t-SNE |
| Evaluation | Silhouette Score, visual assessment |
| Critical model interpretation | Cluster overlap, representational limitations |

## Limitations

1. **Weak cluster separation:** Low Silhouette Scores limit how strongly the discovered groups can be interpreted.
2. **Synthetic observations:** The substantial synthetic proportion may influence estimated structure and generalisability.
3. **Distance sensitivity:** Scaling, encoding and variable weighting affect distance-based clustering.
4. **No external validation reported:** The project did not establish how well discovered clusters correspond to the original obesity classes or clinically meaningful profiles.
5. **Limited model comparison:** DBSCAN is mentioned during the methodological discussion, but the completed modelling and results reported here concern **K-Means and Agglomerative Clustering**.

## Dataset attribution

UCI Machine Learning Repository: [Estimation of Obesity Levels Based on Eating Habits and Physical Condition](https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition). Review the dataset's current citation and licence guidance before redistribution.

## Project takeaway

This project demonstrates an end-to-end **unsupervised learning workflow**: exploring and preparing mixed-type data, fitting two clustering algorithms, selecting candidate group counts, visualising higher-dimensional patterns and critically evaluating cluster quality. The main conclusion is not that distinct obesity-risk groups have been reliably identified, but that **K-Means yielded a relatively stronger yet still weak grouping under the tested conditions**.

---
