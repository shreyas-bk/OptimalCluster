'OptimalCluster.opticlust.Optimal'
To find optimal number of clusters using different 
    optimal clustering algorithms such as elbow, elbow-k_factor, 
    silhouette, gap statistics, gap statistics with standard 
    error, and gap statistics without log.
    
    Parameters
    
    ----------
    
    kmeans_kwargs : dict
        Arguements to be passed to the KMeans algorithm 
        which will be used to determine the optimal number of clusters.
    
    ----------
    
    Methods
    
    ----------
    
    elbow : Determines optimal number of clusters using elbow method. 
    
        Wikipedia - In cluster analysis, the elbow method is a heuristic 
        used in determining the number of clusters in a data set. The 
        method consists of plotting the explained variation as a function
        of the number of clusters, and picking the elbow of the curve as 
        the number of clusters to use. 
        https://en.wikipedia.org/wiki/Elbow_method_(clustering).
        
        The variations are plotted as a scree plot, and the optimal cluster
        value is taken as either the inflection point (where the angle of the 
        scree plot changes the most), or the point after which the scree plot 
        becomes linear. Variations can be measures using inertia or distortions.
    
    elbow_kf : Determines optimal number of clusters by measuring linearity 
        along with a k factor analysis.
        
        Uses scree plot as descibed in elbow method. After standardizing to a 
        fixed scale, the linearity of the slopes between the points is 
        caluculated by measuring their relative means and standard deviations. 
        The last point where the mean becomes greater than the product of the 
        standard deviation and the standard error weight (which is called the 
        k criteria), is taken as the optimal number of clusters. k factor is 
        measured by finding the percentage of points before the optimal cluster 
        vaue that satisfy the k criteria. In general, overlapping clusters 
        result in a lower k factor value, but this can be remedied by 
        increasing the standard error weight up until the optimal cluster value 
        does not change.  
        
        For more details visit *link*
        
    silhouette : Determines optimal number of clusters by finding the cluster 
        value that gives the maximum silhouette score (using sklearn's 
        silhouette_score).
        
        For more details on sklearn's silhouette_score visit 
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html
        
        Wikipedia - The silhouette value is a measure of how similar an object 
        is to its own cluster (cohesion) compared to other clusters (separation).
        The silhouette ranges from −1 to +1, where a high value indicates that 
        the object is well matched to its own cluster and poorly matched to 
        neighboring clusters.
        https://en.wikipedia.org/wiki/Silhouette_(clustering)
        
    gap_stat : Determine the optimal number of clusters using the gap statistic 
        as descibed by Tibshirani, Walther and Hastie.
        
        For more details about gap statistic visit
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        
        This method determines the optimal cluster value by finding the 
        cluster that provides the highest gap statistic value.
        
    gap_stat_se : Determine the optimal number of clusters using the gap statistic 
        along with standard error as descibed by Tibshirani, Walther and Hastie.
        
        For more details about gap statistic with standard error visit
        http://www.web.stanford.edu/~hastie/Papers/gap.pdf
        
        The gap_stat method is not always accurate in cases where clusters are not 
        well defined. This method applies a standard error approach where the first 
        point that satisfies the standard error condition is taken to be the optimal 
        cluster value. 
        
    gap_stat_wolog : Determines the optimal cluster value in a similar fashion to 
        the gap statistic method, but without caluclating logarithms as descirbed by 
        Mohajer, Englmeier and Schmid.
        https://core.ac.uk/reader/12172514
        
    ----------
    
    Example:
    >>> from OptimalCluster.opticlust import Optimal
    >>> opt = Optimal({'max_iter':200})
    
    ----------
    
'Optimal.elbow'
Determines optimal number of clusters using elbow method.
        
         Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
        
        function : {'inertia','distortion'}, default = 'inertia'
            The function used to calculate Variation
            
            'inertia' : Variation used in KMeans is inertia
            
            'distortion' : Variation used in KMeans is distortion
            
            For more information visit
            https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
            
        method : {'angle','lin'}, default = 'angle'
            The method used to calculate the optimal cluster value.
            
            'angle' : The point which has the largest angle of inflection is taken as 
                the optimal cluster value. Works well for lesser number of actual clusters.
                
            'lin' : The first point where the sum of the squares of the difference in 
                slopes of every point after it is less than the sq_er parameter is 
                taken as the optimal cluster value. Works well for a larger range of 
                actual cluster values since the sq_er parameter is adjustable. 
                For more details visit *link*
                
        sq_er : int, default = 1
            Only used when method parameter is 'lin'. The first point where the sum 
            of the squares of the difference in slopes of every point after it is 
            less than the sq_er parameter is taken as the optimal cluster value. When 
            there is suspected to be overlapping/not well defined clusters, a lower 
            value of this parameter can help separate these clusters and give a better 
            value for optimal cluster.
            For more details visit *link*
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=2, centers=3)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.elbow(df)
        Optimal number of clusters is:  3  
    
        ----------
    
'Optimal.elbow_kf'
Determines optimal number of clusters by measuring linearity 
        along with a k factor analysis.
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
        
        function : {'inertia','distortion'}, default = 'inertia'
            The function used to calculate Variation
            
            'inertia' : Variation used in KMeans is inertia
            
            'distortion' : Variation used in KMeans is distortion
            
            For more information visit
            https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
               
        se_weight : int, default = 0
            The standard error weight parameter used to find the k criteria. In general,
            this parameter is to be increased until a satisfactory k factor is reached, 
            and the corresponding value for cluster number is the optimal cluster. When 
            there are overlapping/not well defined clusters it is better to increase the 
            value of this parameter.
            Note - the value of se_weight will automatically increase by increments of 0.5
            if the k criteria is not satisfied in that iteration.
            For more details visit *link*
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=2, centers=5)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.elbow_kf(df)
        Optimal number of clusters is:  5  with k_factor: 0.6 .
    
        ----------
    
'Optimal.silhouette'
Determines optimal number of clusters by finding the cluster 
        value that gives the maximum silhouette score (using sklearn's 
        silhouette_score).
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        lower : int, default = 2
            Lower limit of cluster number to be checked (inclusive).
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=3, centers=6)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.silhouette(df)
        Optimal number of clusters is:  6
    
        ----------
    
'Optimal.gap_stat'
Determine the optimal number of clusters using the gap statistic 
        as descibed by Tibshirani, Walther and Hastie. Finds optimal 
        cluster value by finding clutser number that produces maximum 
        value of gap statistic.
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        B : int, default = 5
            The number of samples references. 
            For more details about B parameter visit
            http://www.web.stanford.edu/~hastie/Papers/gap.pdf
            
        lower : int, default = 1
            Lower limit of cluster number to be checked (inclusive).
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=3, centers=6)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.gap_stat(df)
        6
    
        ----------

'Optimal.gap_stat_se'
Determine the optimal number of clusters using the gap statistic 
        along with standard error as descibed by Tibshirani, Walther and Hastie.
        This method works better than gap_stat when there are custers that are 
        not well defined.
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        B : int, default = 5
            The number of samples references. 
            For more details about B parameter visit
            http://www.web.stanford.edu/~hastie/Papers/gap.pdf
            
        lower : int, default = 1
            Lower limit of cluster number to be checked (inclusive).
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
            
        se_weight : int, default = 1
            The standard error weight to be multiplied with the standard error. Although
            Tibshirani, Walther and Hastie descibe a '1 - standard error' approach, this
            weight can be adjusted. Generally, when then there are overapping/not-well 
            defined clusters, an increase in this parameter can give a more accurate 
            optimal cluster value.
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=3, centers=6)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.gap_stat_se(df)
        6
    
        ----------

'Optimal.gap_stat_wolog'
Determines the optimal cluster value in a similar fashion to 
        the gap statistic method, but without caluclating logarithms as descirbed
        by Mohajer, Englmeier and Schmid. This method may work better when the 
        gap_stat method tends to overestimate.
        
        Parameters
    
        ----------
    
        df : pandas DataFrame (ideally)
            DataFrame with n features upon which the optimal cluster value needs 
            to be determined.
            
        B : int, default = 5
            The number of samples references. 
            For more details about B parameter visit
            http://www.web.stanford.edu/~hastie/Papers/gap.pdf
            
        upper : int, default = 15
            Upper limit of cluster number to be checked (exclusive).
            
        display : boolean, default = False
            If True then a matplotlib plot is displayed. It contains the scree 
            plot with the inertia/distortion values (standardized to a fixed value) 
            on the Y-axis and the corresponding cluster number on the X-axis.
            
        visualize : boolean, default = False
            If True then the _visualize method is called.
            
        method : {'angle','lin','max'}, default = 'lin'
            The method used to calculate the optimal cluster value.
            
            'angle' : The point which has the largest angle of inflection is taken as 
                the optimal cluster value. Works well for lesser number of actual clusters.
                
            'lin' : The first point where the sum of the squares of the difference in 
                slopes of every point after it is less than the sq_er parameter is 
                taken as the optimal cluster value. Works well for a larger range of 
                actual cluster values since the sq_er parameter is adjustable. 
                For more details visit *link*
                
            'max' : max value (not recommended)
            
        se_weight : int, default = 1
            The standard error weight to be multiplied with the standard error. Although
            Tibshirani, Walther and Hastie descibe a '1 - standard error' approach, this
            weight can be adjusted. Generally, when then there are overapping/not-well 
            defined clusters, an increase in this parameter can give a more accurate 
            optimal cluster value.
            
        sq_er : int, default = 1
            Only used when method parameter is 'lin'. The first point where the sum 
            of the squares of the difference in slopes of every point after it is 
            less than the sq_er parameter is taken as the optimal cluster value. When 
            there is suspected to be overlapping/not well defined clusters, a lower 
            value of this parameter can help separate these clusters and give a better 
            value for optimal cluster.
            For more details visit *link*
            
        ----------
        
        Example:
        >>> from OptimalCluster.opticlust import Optimal
        >>> opt = Optimal()
        >>> from sklearn.datasets.samples_generator import make_blobs
        >>> x, y = make_blobs(1000, n_features=3, centers=6)
        >>> df = pd.DataFrame(x)
        >>> opt_value = opt.gap_stat_wolog(df)
        6
    
        ----------
    
