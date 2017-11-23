This repository contains some code related to fuzzy clustering. So far it has:

* Fuzzy C-Means (FCM)
* A benchmark of various different methods to calculate covariance matrices
* Various cluster validity indices
    * Partition Coefficient (PC)
    * Normalized Partition Coefficient (NPC)
    * Fuzzy Hypervolume (FHV)
    * Fukuyama-Sugeno Index (FS)
    * Xie-Beni Index (XB)
    * Beringer-Hullermeier Index (BH)
    * Bouguessa-Wang-Sun index (BWS)

Here is a plot of some data clustered with FCM:

![data plot](https://github.com/99991/FuzzyClustering/blob/master/plot.png)

And here is a plot of the various cluster validity indices for that data clustered with different number of clusters:

![cluster validity indices plots](https://github.com/99991/FuzzyClustering/raw/master/cvi.png)
