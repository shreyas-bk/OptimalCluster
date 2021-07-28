# OptimalCluster

[![Downloads](https://pepy.tech/badge/optimalcluster)](https://pepy.tech/project/optimalcluster)
(From [PePy](https://pepy.tech/project/OptimalCluster))

Check out https://shreyas-bk.github.io/OptimalCluster/

OptimalCluster is the Python implementation of various algorithms to find the optimal number of clusters. The algorithms include elbow, elbow-k_factor, silhouette, gap statistics, gap statistics with standard error, and gap statistics without log. Various types of visualizations are also supported.

For references about the different algorithms visit the following sites:

elbow : [Elbow Method](https://en.wikipedia.org/wiki/Elbow_method_(clustering).)

elbow_kf : _my own implementation, details to be added soon_

silhouette : [Silhouette Score](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html)

gap_stat : [Paper](http://www.web.stanford.edu/~hastie/Papers/gap.pdf)  |  [Python](https://anaconda.org/milesgranger/gap-statistic/notebook)

gap_stat_wolog : [Paper](https://core.ac.uk/reader/12172514)

# Installation

To install the OptimalCluster package through the Python Package index (PyPI) run the following command:
```
pip install OptimalCluster
```

## Documentation

Visit this link : [Documentation](https://github.com/shreyas-bk/OptimalCluster/blob/master/Documentation.md)

## Example

Visit this link : [Example](https://colab.research.google.com/github/shreyas-bk/OptimalClusterExampleNB/blob/master/Example.ipynb)

_Contributions are welcome, please raise a PR_

# TODO

 - add increment_step param to elbow_kf with default as 0.5
 - New verbose parameter addition for methods
 - Needs checks for upper and lower parameters
