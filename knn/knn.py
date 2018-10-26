import numpy as np 
import pandas as pd 
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
ratings = pd.read_csv("Rating Zeros.csv", index_col=0)
items = pd.read_csv("Items.csv")

#print 3 rows from head
#print(ratings.head(3))

#sklearn KMeans 
kmeans = KMeans(n_clusters=5)
kmeans.fit(ratings)
labels = kmeans.labels_ #label [0-4] for each user 943 total
centroids = kmeans.cluster_centers_ #coordinates of cluster centers array,[n clusters, m movies]
pd.set_option('display.max_rows', ratings.describe().shape[1])
pd.DataFrame(centroids)

#separating movienames and genre types 
movie_names = items.Name
genre_types = items.columns[1:]
items_array = np.array(items)
movie_genre = np.array(items[genre_types])

#print(movie_names)


def FiveStarMovies(centroids, labels, k, NTerms): 
    """returns arrays of five star movies and cluster movies.
        centroid is the center of each cluster,
        labels = label(class) for all the users
        k = number of clusters
        nTerms = ??"""

    five_star_movies = [] #contains 5 * movies for each cluster
    cluster_movies = [] #contains all the movies in each clusters

    for i in range(k): 
        cluster_members = ratings[labels==i] 
        print 'cluster',i+1,'size: ', cluster_members.shape[0] 
        movie_total_rating = np.array(cluster_members.sum(0)).astype(float) 
        movie_rating_count = np.array((cluster_members!=0).sum(0)).astype(float) 
        movie_average_rating = np.divide(movie_total_rating, movie_rating_count) 
        movie_average_rating[np.isnan(movie_average_rating)] = 0 
        rated_movie_list = sorted(zip(movie_names, movie_average_rating), key=lambda x: int(x[1]), reverse=True) 
        to_append = [] 
        cluster_append = [] 
        for j in rated_movie_list: 
            if j[1] == 5: 
                to_append.append(j[0])  
            if j[1] != 0: 
                cluster_append.append(j[0]) 
        five_star_movies.append(to_append) 
        cluster_movies.append(cluster_append) 
        print() 
    # most rated movies in each cluster 
    k = 0 
    for m in five_star_movies: 
        prototype = list(np.zeros(len(genre_types))) 
        for n in items_array:     
            for i in m: 
                if i == n[0]: 
                    prototype += n[1:] 
        print("Representative Movies:")
        for i in range(5): 
            print m[i] 
        print("") 
        prototype = map(int, prototype) 
        movie_genre_to_use = sorted(zip(genre_types, prototype), key=lambda x:int(x[1]), reverse=True) 
        print ("Cluster", k+1, "top genre: ") 
        for i in movie_genre_to_use[:NTerms]: 
            print i[0] 
        print("") 
        k += 1 
    return five_star_movies, cluster_movies

#stores 5 rated movies for each cluster and all the movies in all the clusters
five_star_movies, cluster_movies = FiveStarMovies(centroids, labels, 5,5)


labels = kmeans.labels_; labels[0:5]


#KNN using cluster labels

from sklearn.cross_validation import train_test_split
r_train, r_test, target_train, target_test = train_test_split(ratings, labels, test_size = 0.2, random_state=33)

#print (r_train.shape, r_test.shape)
#print(r_test[0:5])

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler().fit(r_train)
train_norm = min_max_scaler.fit_transform(r_test)
test_norm = min_max_scaler.fit_transform(r_test)



def KNN_Classifier(instance, dat, label, k, measure):
    """ KNN using cluster labels"""

    if measure =="euclidian":
        dists = np.sqrt((dat-instance**2).sum(axis=1))
    elif measure == "cosine":
        dat_norm = np.arrray([np.linalg.norm(dat[j]) for j in range(len(dat))])
        instance_norm = np.lialg.norm(instance)
        sims = np.dot(dat,instance)/(dat_norm*instance_norm)
        dists = 1 - sims
    idx = np.argsort(dists)
    neighbor_index = idx[:k]
    neighbor_record = dat[[neighbor_index]]
    labels = label[[neighbor_index]]
    final_class = np.bincount(labels)
    return np.argmax(final_class), idx[:k]

def Comp_Accuracy(testdata, testlabel, traindata, trainlabel, k,measure):
    """returns the accuracy rate"""

    correct = 0
    for i in range(testdata.shape[0]):
        pred_class = KNN_Classifier(testdata[i],traindata, trainlabel, k, measure)
        if pred_class[0] == testlabel[i]:
            correct += 1
    accuracy_rate = float(correct)/float(testdata.shape[0])
    return accuracy_rate


#compute the accuracy for variable k (1,21)
euc_accuracy = []
'''
for i in range(1,21):
    result = accuracy_score(r_test, target_test)
    abc = euc_accuracy.append(result)
print(abc)
'''
print accuracy_score(test_norm,target_test)

cfm = confusion_matrix(r_test,target_test)

print('Confusion Matrix')
print('\n',cfm)
#for plotting, not working right now
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
print ("Plotting graph: \n")
k = np.array(range(1,21))
euc = np.array(euc_accuracy)
plt.plot(k, euc, 'r--', label='Euclidian Distance')
plt.ylabel('Accuracy rate')
plt.xlabel('Numbre of neighbors (k)')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.show()




def getmovies(instance,dat,label,movies,number,k=1,measure="euclidian"):
    recommendation = []
    assigned_class = KNN_Classifier(instance, dat, label, k, measure)
    for i in range(number):
        recommendation.append(movies[int(assigned_class[0])][i])
    return recommendation


   
recommendation = getmovies(test_norm[0],train_norm,target_train,five_star_movies,5)
for i in range(5):
    print(recommendation[i])






    































