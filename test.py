import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

df=pd.read_csv("datasets/movie_dataset.csv")
data=df.filter(['index','title','genres'], axis=1)
# print(data.head(3))
features=['index','title','genres']
for feature in features:
    data[feature] = data[feature].fillna('')
# print(data.isnull().any())

x = data.genres
a = list()
for i in x:
    abc = i
    a.append(abc.split(' '))
a = pd.DataFrame(a) 
# print(a[0].unique())
b = a[0].unique()
for i in b:
    data[i] = 0
for i in b:
     data.loc[data['genres'].str.contains(i), i] = 1

data = data.drop(['index','genres','title'],axis =1)
kmeanModel = KMeans(n_clusters=8)
kmeanModel.fit(data)
data['Cluster'] = kmeanModel.labels_
# print(data.head(3))
# print(kmeanModel.labels_)
def get_title_from_index(index):
    # print(index)
    return df[df['index']==index]["title"].values[0]
def get_index_from_title(title):
    # print(title)
    if(len(df[df['title']==title]["index"].values)==0): return -1
    return df[df['title']==title]["index"].values[0]
def get_director_from_index(index):
    # print(index)
    return df[df['index']==index]["director"].values[0]
def get_keywords_from_index(index):
    # print(index)
    return df[df['index']==index]["keywords"].values[0]
def get_cast_from_index(index):
    # print(index)
    return df[df['index']==index]["cast"].values[0]

movie_user_likes = "Av"
movie_index = get_index_from_title(movie_user_likes)

print(df[df.index==movie_index].overview[0])
# if(movie_index==-1): print("not in cluster")
# d=[[0,movie_user_likes,get_director_from_index(movie_index),get_cast_from_index(movie_index),get_keywords_from_index(movie_index)]]
# new_df=pd.DataFrame(d, columns=['index','title','director','cast','keywords'])
# # print(d)
# flag=data['Cluster'][movie_index]
# # print(flag)

# j=1
# for i in data['Cluster']:
#     if i==flag:
#         row={'index':j,'title':get_title_from_index(j), 'director':get_director_from_index(j), 'cast':get_cast_from_index(j), 'keywords':get_keywords_from_index(j)}
#         new_df = new_df.append(row, ignore_index=True)  
#         j=j+1
# # print(new_df.head(4))
# features = ['keywords','cast','director']
# for feature in features:
#     new_df[feature] = new_df[feature].fillna('') 

# def combine_features(row):
#     return row['keywords']+" "+row['cast']+" "+row['director']

# new_df["combined_features"] = new_df.apply(combine_features,axis=1) 
# cv = CountVectorizer() 
# count_matrix = cv.fit_transform(new_df["combined_features"])


# cosine_sim = cosine_similarity(count_matrix)
# similar_movies = list(enumerate(cosine_sim[0]))
# # print(similar_movies)
# sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]

# def get_title_from_index2(index):
#     return new_df[new_df['index']==index]["title"].values[0]
# def get_director_from_index2(index):
#     # print(index)
#     return new_df[new_df['index']==index]["director"].values[0]
# def get_keywords_from_index2(index):
#     # print(index)
#     return new_df[new_df['index']==index]["keywords"].values[0]
# def get_cast_from_index2(index):
#     # print(index)
#     return new_df[new_df['index']==index]["cast"].values[0]
# # print(new_df.columns)
# i=0
# movie_name=[]
# dir_name=[]
# keywords_name=[]
# cast_name=[]
# print("Top 5 similar movies to "+movie_user_likes+" are:\n")
# for element in sorted_similar_movies:
#     #print(get_title_from_index2(element[0]))
#     movie_name.append(get_title_from_index2(element[0]))
#     dir_name.append(get_director_from_index2(element[0]))
#     keywords_name.append(get_keywords_from_index2(element[0]))
#     cast_name.append(get_cast_from_index2(element[0]))
#     i=i+1
#     if i>5:
#         break
# print(movie_name)
# movie_name1=movie_name[0]
# movie_name2=movie_name[1]
# movie_name3=movie_name[2]
# movie_name4=movie_name[3]
# movie_name5=movie_name[4]