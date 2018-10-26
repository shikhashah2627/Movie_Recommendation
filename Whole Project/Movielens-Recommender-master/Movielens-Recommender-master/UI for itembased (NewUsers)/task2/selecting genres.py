import csv 
import numpy as np
import pandas as pd

f = open("Items.csv")
items = pd.read_csv("Items.csv", index_col=0)
movie_names = items.columns[1:]
movie_with_genre = []
final_list = ([])
for movie in f:
    final_list.append([])
    movie_name = movie.split(',')[1]
    ratings = movie.split(',')[2:]
    '''
    print "movie is"
    print movie_name
    print "ratings"
    print ratings
    '''
    ratings_max = np.argmax(ratings)
    if ratings_max == 19 : ratings_max = ratings_max - 1
    get_genre = movie_names[ratings_max]
    movie_with_genre.append(get_genre)
    print get_genre
    final_list[movie_name].append(get_genre)
    print final_list