#requirements: numpy, pandas, sklearn, scipy, matplotlib, mpl_toolkits, seaborn
import numpy as np
import pandas as pd # added pandas module for v 0.0.5 (needed when features=3, visualize=True)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import warnings #v0.0.6 added warnings

class Optimal():#v0.1.0 added documentation and comments
    
    """
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
    
    """
    
    opti_df = None #check TODO
    
    def __init__(
        self,
        kmeans_kwargs: dict = None
    ):
        """
        Construct Optimal with parameter kmeans_kwargs.
        
        """
        self.kmeans_kwargs = kmeans_kwargs #check TODO
    
    def elbow(self,df,upper=15,display=False,visualize=False,function='inertia',method='angle',sq_er=1):
        """
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
    
        """
        
        #lower is always 1, list to store inertias
        lower=1
        inertia = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        #populating inertia
        for i in K:
            
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)

            if function=='inertia':
                
                #appending KMeans inertia_ variable
                inertia.append(cls.inertia_)
            elif function=='distortion':
                
                #distortion value from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
                inertia.append(sum(
                                np.min(
                                    cdist(df, cls.cluster_centers_, 'euclidean'),axis=1
                                       )
                                    ) / df.shape[0])
            else:
                
                #for incorrect function
                raise ValueError('function should be "inertia" or "distortion"')
        
        #standardizing inertia to a fixed range
        inertia = np.array(inertia)/(np.array(inertia)).max()*14 #v0.0.6 changed to fixed number of 14 (elbow)
        
        #calulating slopes
        slopes = [inertia[0]-inertia[1]]#check TODO
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
            
        #calculating angles
        angles = []
        for i in range(len(slopes)-1):
            angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
            
        #plotting scree plot
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
            
        #finding optimal cluster value
        extra=''
        
        #using maximum angle method
        if method == 'angle':
            optimal = np.array(angles).argmax()+1
            confidence = round(np.array(angles).max()/90*100,2)#percentage of angle out of 90
            if confidence<=50:
                extra=' with Confidence:'+str(confidence)+'%.'+' Try using elbow_kf, gap_stat_se or other methods, or change the method parameter to "lin"'
        
        #using linearity method
        elif method == 'lin': #v0.0.6 changed method for lin
            flag=False
            for i in range(len(slopes)-1):
                
                #finding first point that satisfies condition
                if (sum([(slopes[i]-slopes[j])**2 for j in range(i+1,len(slopes))]))<=sq_er:
                    optimal = i
                    flag=True
                    break
            #if no point satisfies, raise warning
            if flag==False:
                optimal=upper-1
                warnings.warn("Optimal cluster value did not satisfy sq_er condition. Try increasing value of parameter upper for a better result")
        
        #for incorrect method
        else:
            raise ValueError('method should be "angle" or "lin"')
            
        #calling visualization    
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
                # raised ValueErrors for fail cases v 0.0.5
        
        #printing result
        print('Optimal number of clusters is: ',str(optimal),extra) 
        
        #return optimal cluster value
        return optimal 
    
    def elbow_kf(self,df,upper=15,display=False,visualize=False,function='inertia',se_weight=0):#v0.1.0 changed default value to 0 for se_weight param
        """
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
    
        """
        
        #lower is always 1, list to store inertias
        lower=1
        inertia = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        #populating inertia
        for i in K:
            
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            
            if function=='inertia':
                
                #appending KMeans inertia_ variable
                inertia.append(cls.inertia_)
            elif function=='distortion':
                
                #distortion value from https://www.geeksforgeeks.org/elbow-method-for-optimal-value-of-k-in-kmeans/
                inertia.append(sum(
                                np.min(
                                    cdist(df, cls.cluster_centers_, 'euclidean'),axis=1
                                       )
                                    ) / df.shape[0])
            else:
                
                #for incorrect function
                raise ValueError('function should be "inertia" or "distortion"')
                
        #standardizing inertia to a fixed range
        inertia = np.array(inertia)/(np.array(inertia)).max()*14
        
        #calulating slopes
        slopes = [inertia[0]-inertia[1]]
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
            
        #plotting scree plot
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
            
        #making lists for means and standard deviations
        means = []
        sds = []
        
        #populating means and sds
        for i in range(len(slopes)-1):
            means.append(np.mean(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
            sds.append(np.std(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
            
        #increment se_weight until k criteria is fulfilled
        flag=False
        while(not flag):
            diffs = [x[0]-se_weight*x[1]>0 for x in zip(means,sds)]
            optimal = (len(diffs) - list(reversed(diffs)).index(False)) if False in diffs else -1
            if optimal==-1:
                se_weight+=0.5
                if se_weight>15:#v0.0.6 TODO: find a better way to check this fail case
                    warnings.warn('Optimal cluster not found even with high se_weight, try increasing upper parameter')
                    return -1
            else:
                flag=True
                
        #find k_factor as percent of points before optimal value that satisfy k criteria
        k_factor = (round(diffs[0:optimal].count(False)/len(diffs[0:optimal]),2))
        
        #calling visualization  
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        
        #printing result
        extra = '' if k_factor>=0.60 else 'Lesser k_factor may be due to overlapping clusters, try increasing the se_weight parameter to '+str(se_weight+0.5)
        print('Optimal number of clusters is: ',str(optimal),' with k_factor:',str(k_factor),'.',extra)
        
        #returning optimal cluster value
        return optimal
    
    def silhouette(self,df,lower=2,upper=15,display=False,visualize=False):#v0.0.6 added slihouette method
        """
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
    
        """
        
        #sklearn's silhuoette score
        from sklearn.metrics import silhouette_score
        
        #list for silhuoette scores
        scores = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        for i in K:
            
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            
            #appending scores
            scores.append(silhouette_score(df,cls_assignment))
            
        #plotting scree plot
        if display==True:
            plt.plot(K, scores, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('silhouette_score') 
            plt.title('Silhouette Analysis') 
            plt.show()
            
        #getting optimal cluster value with max score
        optimal = K[scores.index(sorted(scores,reverse=True)[0])]
        
        #calling visualization  
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        
        #printing result
        print('Optimal number of clusters is: ',str(optimal))
        
        #returning optimal cluster value
        return optimal
    
    def gap_stat(self,df,B=5,lower=1,upper=15,display=False,visualize=False):#v0.0.6 added gap_stat method
        """
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
    
        """
        
        #warning when sample size is large
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
            
        #list for gap statistic values
        gaps = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        for i in K:
            
            #list for inertias of sample datasets
            W_star = []
            
            for k in range(B):
                
                #fitting KMeans with i value for n_clusters for sample dataset
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
            
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            
            #appending gap value
            W = cls.inertia_
            gaps.append(np.mean(np.log(W_star)) - np.log(W))
            
        #plotting scree plot
        if display==True:
            plt.plot(K, gaps, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.title('Gaps Statistic Analysis') 
            plt.show()
            
        #optimal cluster value is when gap stat is max
        optimal = np.array(gaps).argmax()+1
        
        #warning if K range may be too small
        if optimal==len(gaps):
            warnings.warn('Try increasing upper parameter for a better result.')
            
        #calling visualization  
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        
        #returning optimal cluster value
        return optimal
    
    def gap_stat_se(self,df,B=5,lower=1,upper=15,display=False,visualize=False,se_weight=1):
        """
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
    
        """
        import math
        
        #warning when sample size is large
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
            
        #lists for gap statistic values and s values
        gaps = []
        s = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        for i in K:
            
            #list for inertias of sample datasets
            W_star = []
            
            for k in range(B):
                
                #fitting KMeans with i value for n_clusters for sample dataset
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
                
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            
            #appending gap and s values
            W = cls.inertia_
            gaps.append(np.mean(np.log(W_star)) - np.log(W))
            sd = np.std(np.log(W_star))
            s.append(math.sqrt(1+(1/B))*sd)
            
        #plotting scree plot
        if display==True:
            plt.plot(K, gaps, 'bx-',label='gaps') 
            diff = np.append((np.array(gaps)-se_weight*np.array(s))[1:],(np.array(gaps)-se_weight*np.array(s))[-1:])
            plt.plot(K, diff, 'rx-',label='diff')
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.legend()
            plt.title('Gaps Statistic Analysis') 
            plt.show()
            
        #checking if optimal cluster is found, else raise warning 
        flag = False
        for i in range(0,len(gaps)-1):
            if (gaps[i]>=gaps[i+1]-se_weight*s[i+1]):
                optimal = i+1
                flag = True
                break
        if flag==False:
            warnings.warn('Could not find an optimal point using this method, optimal cluster returned is from gap_stat method. Try increasing upper parameter')
            optimal = np.array(gaps).argmax()+1#v0.0.6 added fail check for gap_stat_se
            
        #calling visualization  
        if visualize==True:
            x = self._visualization(df,optimal) 
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')
        
        #returning optimal cluster value
        return optimal
    
    def gap_stat_wolog(self,df,B=5,upper=15,display=False,visualize=False,method='lin',se_weight=1,sq_er=1):
        """
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
    
        """
        
        #warning when sample size is large
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
            
        #lower is always 1
        lower = 1
        
        #list for gap statistic values
        gaps = []
        
        #K is the range of cluster values to be checked
        K=range(lower,upper)
        
        for i in K:
            
            #list for inertias of sample datasets
            W_star = []
            
            for k in range(B):
                
                #fitting KMeans with i value for n_clusters for sample dataset
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
                
            #fitting KMeans with i value for n_clusters
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            
            #appending gap value (without log)
            W = cls.inertia_
            gaps.append(np.mean((W_star)) - (W))
            
        #standardizing
        gaps = (np.array(gaps)-np.array(gaps).min())/(np.array(gaps).max()-np.array(gaps).min())*14#v0.0.6 changed to fixed number of 14 (elbow)
        
        #list for slopes
        slopes = [gaps[0]-gaps[1]]
        for i in range(len(gaps)-1):
            slopes.append((gaps[i+1]-gaps[i]))
            
        #finding optimal cluster value
        if method=='angle':
            
            #finding cluster value with max angle between slopes
            angles = []
            for i in range(len(slopes)-1):
                angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
            optimal = np.array(angles).argmax()+1
            
        elif method=='lin':#v0.0.6 changed algo for lin method for gap_stat_wolog
            
            #Using linearity to find optimal clulster value
            flag=False
            for i in range(len(slopes)-1):
                if (sum([(slopes[i]-slopes[j])**2 for j in range(i+1,len(slopes))]))<=sq_er:
                    optimal = i
                    flag=True
                    break
            if flag==False:
                optimal=upper-1
                warnings.warn("Optimal cluster value did not satisfy sq_er condition. Try increasing value of parameter upper for a better result")
        
        elif method=='max':
            optimal = np.array(gaps).argmax()+1
        
        else:
            
            #for incorrect method
            raise ValueError('method should be "lin","angle" or "max"')
            
        #plotting scree plot
        if display==True:
            plt.plot(K, gaps, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.title('Gaps Statistic Analysis') 
            plt.show()
            
        #calling visualization  
        if visualize==True:
            x = self._visualization(df,optimal)
            
            #warning when feature size is incorrect
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')
        
        #returning optimal cluster value
        return optimal

    def _visualization(self,df,optimal):#v0.0.6 made visualization method a class method
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        
        #fitting KMeans with optimal cluster value
        cls = KMeans(n_clusters=optimal,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=optimal)
        cls_assignment = cls.fit_predict(df)
        
        
        if len(df.columns) == 1:
            
            #visualization with stripplot for 1 feature
            col_name = df.columns[0]
            sns.stripplot(data = df,x=['']*len(df),y=col_name,hue=cls_assignment)
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
            
        elif len(df.columns)==2:
            
            #visualization with scatterplot for 2 features
            col_name1 = df.columns[0]
            col_name2 = df.columns[1]
            sns.scatterplot(data=df,x=col_name1,y=col_name2,hue=cls_assignment,palette='Set1')
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
            
        elif len(df.columns)==3:
            
            #visualization with 3D plot for 3 features
            fig = plt.figure()
            ax = plt.axes(projection="3d")
            col_name1 = df.columns[0]
            col_name2 = df.columns[1]
            col_name3 = df.columns[2]
            ax.scatter3D(xs=df[col_name1],ys=df[col_name2],zs=df[col_name3],c=pd.Series(cls_assignment))
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        else:
            return 'fail'
    
    
        #TODO : add checks for upper,lower, other params
        #TODO : for optimal cluster of 1
        #TODO : opti_df
        #TODO : add verbose param
        #TODO : optimize constructor for kmeans_kwargs
        #TODO : first value of slopes
