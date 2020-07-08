import numpy as np
import pandas as pd # added pandas module for v 0.0.5 (needed when features=3, visualize=True)
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import seaborn as sns
import warnings #v0.0.6 added warnings

class Optimal():
    
    """
    To find optimal number of clusters using different optimal clustering algorithms
    *citation*
    *parameters*
    *methods*
    *example*
    """
    
    opti_df = None
    
    def __init__(
        self,
        kmeans_kwargs: dict = None
    ):
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        self.kmeans_kwargs = kmeans_kwargs
    
    def elbow(self,df,upper=15,display=False,visualize=False,function='inertia',method='angle',sq_er=1):
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        lower=1
        inertia = []
        K=range(lower,upper)
        for i in K:
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            if function=='inertia':
                inertia.append(cls.inertia_)
            elif function=='distortion':
                inertia.append(sum(np.min(cdist(df, cls.cluster_centers_, 
                          'euclidean'),axis=1)) / df.shape[0])
            else:
                raise ValueError('function should be "inertia" or "distortion"')
        inertia = np.array(inertia)/(np.array(inertia)).max()*14 #v0.0.6 changed to fixed number of 14 (elbow)
        slopes = [inertia[0]-inertia[1]]
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
        angles = []
        for i in range(len(slopes)-1):
            angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
        extra=''
        if method == 'angle':
            optimal = np.array(angles).argmax()+1
            confidence = round(np.array(angles).max()/90*100,2)
            if confidence<=50:
                extra=' with Confidence:'+str(confidence)+'%.'+' Try using elbow_kf, gap_stat_se or other methods, or change the method parameter to "lin"'
        elif method == 'lin': #v0.0.6 changed method for lin
            flag=False
            for i in range(len(slopes)-1):
                if (sum([(slopes[i]-slopes[j])**2 for j in range(i+1,len(slopes))]))<=sq_er:
                    optimal = i
                    flag=True
                    break
            if flag==False:
                optimal=upper-1
                warnings.warn("Optimal cluster value did not satisfy sq_er condition. Try increasing value of parameter upper for a better result")
        else:
            raise ValueError('method should be "angle" or "lin"')
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        print('Optimal number of clusters is: ',str(optimal),extra) # raised ValueErrors for fail cases v 0.0.5
        return optimal 
    
    def elbow_kf(self,df,upper=15,display=False,visualize=False,function='inertia',se_weight=1.5):#v0.0.6 added elbow_kf method
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        lower=1
        inertia = []
        K=range(lower,upper)
        for i in K:
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            if function=='inertia':
                inertia.append(cls.inertia_)
            elif function=='distortion':
                inertia.append(sum(np.min(cdist(df, cls.cluster_centers_, 
                          'euclidean'),axis=1)) / df.shape[0])
            else:
                raise ValueError('function should be "inertia" or "distortion"')
        inertia = np.array(inertia)/(np.array(inertia)).max()*14
        slopes = [inertia[0]-inertia[1]]
        for i in range(len(inertia)-1):
            slopes.append(-(inertia[i+1]-inertia[i]))
        if display==True:
            plt.plot(K, inertia, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel(function) 
            plt.title('The Elbow Method using '+function) 
            plt.show()
        means = []
        sds = []
        for i in range(len(slopes)-1):
            means.append(np.mean(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
            sds.append(np.std(np.array(slopes[i:i+3 if i+3<len(slopes) else (i+2 if i+2<len(slopes) else i+1)])))
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
        k_factor = (round(diffs[0:optimal].count(False)/len(diffs[0:optimal]),2))
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        extra = '' if k_factor>=0.60 else 'Lesser k_factor may be due to overlapping clusters, try increasing the se_weight parameter to '+str(se_weight+0.5)
        print('Optimal number of clusters is: ',str(optimal),' with k_factor:',str(k_factor),'.',extra)
        return optimal
    
    def silhouette(self,df,lower=2,upper=15,display=False,visualize=False):#v0.0.6 added slihouette method
        from sklearn.metrics import silhouette_score
        scores = []
        K=range(lower,upper)
        for i in K:
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            scores.append(silhouette_score(df,cls_assignment))
        if display==True:
            plt.plot(K, scores, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('silhouette_score') 
            plt.title('Silhouette Analysis') 
            plt.show()
        optimal = K[scores.index(sorted(scores,reverse=True)[0])]
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        print('Optimal number of clusters is: ',str(optimal))
        return optimal
    
    def gap_stat(self,df,B=5,lower=1,upper=15,display=False,visualize=False):#v0.0.6 added gap_stat method
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
        gaps = []
        K=range(lower,upper)
        for i in K:
            W_star = []
            for k in range(B):
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            W = cls.inertia_
            gaps.append(np.mean(np.log(W_star)) - np.log(W))
        if display==True:
            plt.plot(K, gaps, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.title('Gaps Statistic Analysis') 
            plt.show()
        optimal = np.array(gaps).argmax()+1
        if optimal==len(gaps):
            warnings.warn('Try increasing upper parameter for a better result.')
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')#v0.0.6, changed ValueError to warning
        return optimal
    
    def gap_stat_se(self,df,B=5,lower=1,upper=15,display=False,visualize=False,se_weight=1):
        import math
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
        gaps = []
        s = []
        K=range(lower,upper)
        for i in K:
            W_star = []
            for k in range(B):
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            W = cls.inertia_
            gaps.append(np.mean(np.log(W_star)) - np.log(W))
            sd = np.std(np.log(W_star))
            s.append(math.sqrt(1+(1/B))*sd)
        if display==True:
            plt.plot(K, gaps, 'bx-',label='gaps') 
            diff = np.append((np.array(gaps)-se_weight*np.array(s))[1:],(np.array(gaps)-se_weight*np.array(s))[-1:])
            plt.plot(K, diff, 'rx-',label='diff')
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.legend()
            plt.title('Gaps Statistic Analysis') 
            plt.show()
        flag = False
        for i in range(0,len(gaps)-1):
            if (gaps[i]>=gaps[i+1]-se_weight*s[i+1]):
                optimal = i+1
                flag = True
                break
        if flag==False:
            warnings.warn('Could not find an optimal point using this method, optimal cluster returned is from gap_stat method. Try increasing upper parameter')
            optimal = np.array(gaps).argmax()+1#v0.0.6 added fail check for gap_stat_se
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')
        return optimal
    
    def gap_stat_wolog(self,df,B=5,upper=15,display=False,visualize=False,method='lin',se_weight=1,sq_er=1):
        if B*len(df)*upper>3000:
            warnings.warn('Many cases to check, may take some time')
        lower = 1
        gaps = []
        K=range(lower,upper)
        for i in K:
            W_star = []
            for k in range(B):
                sample = np.random.random_sample(size=df.shape)
                W_star.append(KMeans(i).fit(sample).inertia_)
            cls = KMeans(n_clusters=i,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=i)
            cls_assignment = cls.fit_predict(df)
            W = cls.inertia_
            gaps.append(np.mean((W_star)) - (W))
        gaps = (np.array(gaps)-np.array(gaps).min())/(np.array(gaps).max()-np.array(gaps).min())*14#v0.0.6 changed to fixed number of 14 (elbow)
        slopes = [gaps[0]-gaps[1]]
        for i in range(len(gaps)-1):
            slopes.append((gaps[i+1]-gaps[i]))
        if method=='angle':
            angles = []
            for i in range(len(slopes)-1):
                angles.append(np.degrees(np.arctan((slopes[i]-slopes[i+1])/(1+slopes[i]*slopes[i+1]))))
            optimal = np.array(angles).argmax()+1
        elif method=='lin':#v0.0.6 changed algo for lin method for gap_stat_wolog
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
            raise ValueError('method should be "lin","angle" or "max"')
        if display==True:
            plt.plot(K, gaps, 'bx-') 
            plt.xlabel('Values of K') 
            plt.ylabel('Gaps') 
            plt.title('Gaps Statistic Analysis') 
            plt.show()
        if visualize==True:
            x = self._visualization(df,optimal) 
            if x=='fail':
                warnings.warn('Could not visualize: Number of columns of the DataFrame should be between 1 and 3 for visualization')
        return optimal

    def _visualization(self,df,optimal):#v0.0.6 made visualization method a class method
        """
        *description*
        *citation*
        *parameters*
        *methods*
        *example*
        """
        cls = KMeans(n_clusters=optimal,**self.kmeans_kwargs) if self.kmeans_kwargs is not None else KMeans(n_clusters=optimal)
        cls_assignment = cls.fit_predict(df)
        if len(df.columns) == 1:
            col_name = df.columns[0]
            sns.stripplot(data = df,x=['']*len(df),y=col_name,hue=cls_assignment)
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        elif len(df.columns)==2:
            col_name1 = df.columns[0]
            col_name2 = df.columns[1]
            sns.scatterplot(data=df,x=col_name1,y=col_name2,hue=cls_assignment,palette='Set1')
            plt.title('Clustering with '+str(optimal)+' clusters')
            plt.show()
        elif len(df.columns)==3:
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
    
    
        #TODO: add documentation
        #TODO: add checks for upper,lower, other params
        #TODO: for optimal cluster of 1
