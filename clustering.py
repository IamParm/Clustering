
# %% K-means clustering
# data
from sklearn import datasets

X, y = datasets.make_blobs(
    n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=123
)

# call leaner
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3)
pred = km.fit_predict(X)

# metrics
from sklearn.metrics import homogeneity_score

print(f"homogeneity: {homogeneity_score(y, pred)}")
print(f"inertia: {km.inertia_}")

# %% How to determine the number of cluster
# data
from sklearn import datasets

X, y = datasets.make_blobs(
    n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=123
)
# determine of clusters
from sklearn.cluster import KMeans

inertia, homogeneity = {}, {}
for k in range(1, 10):
    km = KMeans(n_clusters=k)
    pred = km.fit_predict(X)
    inertia[k] = km.inertia_
    homogeneity[k] = homogeneity_score(y, pred)

# plot
import seaborn as sns
import matplotlib.pyplot as plt

ax = sns.lineplot(
    x=list(inertia.keys()),
    y=list(inertia.values()),
    color="blue",
    label="inertia",
    legend=None,
)
ax.set_ylabel("inertia")
ax.twinx()
ax = sns.lineplot(
    x=list(homogeneity.keys()),
    y=list(homogeneity.values()),
    color="red",
    label="homogeneity",
    legend=None,
)
ax.set_ylabel("homogeneity")
ax.figure.legend()

# %% data vs prediction (K-means)
_, axes = plt.subplots(1, 2)
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=y, ax=axes[0])
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=pred, ax=axes[1])

# %% Hierarchical clustering
# data
from sklearn import datasets

X, y = datasets.make_blobs(
    n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=123
)
# call leaner
from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters=3)
pred = ac.fit_predict(X)
## metrics
from sklearn.metrics import homogeneity_score

print(f"homogeneity: {homogeneity_score(y, pred)}")

# %% data vs prediction (Hierarchical clustering)
_, axes = plt.subplots(1, 2)
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=y, ax=axes[0])
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=pred, ax=axes[1])


# %% DBSCAN
# data
from sklearn import datasets

X, y = datasets.make_blobs(
    n_samples=1000, cluster_std=[1.0, 2.5, 0.5], random_state=123
)
# call leaner
from sklearn.cluster import DBSCAN

dbscan = DBSCAN()
pred = dbscan.fit_predict(X)
## metrics
from sklearn.metrics import homogeneity_score

print(f"homogeneity: {homogeneity_score(y, pred)}")

# %% data vs prediction (DBSCAN)
_, axes = plt.subplots(1, 2)
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=y, ax=axes[0])
sns.scatterplot(x=[x[0] for x in X], y=[x[1] for x in X], hue=pred, ax=axes[1])

# %% Working with categoricals
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

enc_ordinal = OrdinalEncoder()
enc_onehot = OneHotEncoder(handle_unknown="ignore")
X = pd.DataFrame({"Sex": ["F", "M", "F", "unknown"], "Height": [173, 174, 175, 176]})
X

# %%
enc_ordinal.fit_transform(X[["Sex"]])

# %%
enc_onehot.fit_transform(X[["Sex"]]).toarray()

# %% Working with missing value
import numpy as np
from sklearn.impute import SimpleImputer

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X = pd.DataFrame({"Sex": ["F", "M", "F", "X"], "Height": [173, np.nan, 175, 176]})
X

# %%
imp.fit_transform(X[["Height"]])

# %% ColumnTransformer
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import Normalizer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

imp = SimpleImputer(missing_values=np.nan, strategy="mean")
enc = OneHotEncoder(handle_unknown="ignore")
X = pd.DataFrame(
    {
        "Grade": ["A", "B", "C", "C"],
        "Sex": ["F", "M", "F", "X"],
        "Height": [173, np.nan, 175, 176],
        "value1": [4, 1, 5, 2],
        "value2": [1, 3, 9, 3],
        "value3": [5, 7, 5, 1],
        "value4": [1122, 1321, 3213, 212],
    }
)
X

# %%
ct = ColumnTransformer(
    [
        ("sex_ohe", enc, ["Sex", "Grade"]),
        ("height_fillna", imp, ["Height"]),
        ("norm1", Normalizer(norm="l2"), ["value1", "value2"]),
        ("norm2", Normalizer(norm="max"), ["value1", "value3"]),
    ],
    remainder="passthrough",
)

# %%
ct.fit_transform(X)

# %% Grid Search and Cross validation
from sklearn import svm, datasets
from sklearn.model_selection import GridSearchCV

iris = datasets.load_iris()
parameters = {"kernel": ("linear", "rbf"), "C": [1, 10]}
svc = svm.SVC()
clf = GridSearchCV(svc, parameters, cv=5)
clf.fit(iris.data, iris.target)
dev_means = clf.cv_results_["mean_test_score"]
dev_stds = clf.cv_results_["std_test_score"]
dev_params = clf.cv_results_["params"]
print("Development Set Results")
for mean, std, params in zip(dev_means, dev_stds, dev_params):
    print(f"  {mean:.3f} +/- {std*2:.3f} for {params}")

# %%