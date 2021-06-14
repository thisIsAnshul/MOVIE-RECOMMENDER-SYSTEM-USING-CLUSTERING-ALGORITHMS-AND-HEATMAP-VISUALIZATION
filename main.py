import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans

import json
import bs4 as bs
import urllib.request
import pickle
import requests

# load the nlp model and tfidf vectorizer from disk
filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
vectorizer = pickle.load(open('tranform.pkl','rb'))

def create_similarity():
    data = pd.read_csv('main_data.csv')
    # creating a count matrix
    cv = CountVectorizer()
    count_matrix = cv.fit_transform(data['comb'])
    # creating a similarity score matrix
    similarity = cosine_similarity(count_matrix)
    return data,similarity

def rcmd(m):
    m = m.lower()
    try:
        data.head()
        similarity.shape
    except:
        data, similarity = create_similarity()
    if m not in data['movie_title'].unique():
        return('Sorry! The movie you requested is not in our database. Please check the spelling or try with some other movies')
    else:
        i = data.loc[data['movie_title']==m].index[0]
        lst = list(enumerate(similarity[i]))
        lst = sorted(lst, key = lambda x:x[1] ,reverse=True)
        lst = lst[1:11] # excluding first item since it is the requested movie itself
        l = []
        for i in range(len(lst)):
            a = lst[i][0]
            l.append(data['movie_title'][a])
        return l
    
# converting list of string to list (eg. "["abc","def"]" to ["abc","def"])
def convert_to_list(my_list):
    my_list = my_list.split('","')
    my_list[0] = my_list[0].replace('["','')
    my_list[-1] = my_list[-1].replace('"]','')
    return my_list

def get_suggestions():
    data = pd.read_csv('main_data.csv')
    return list(data['movie_title'].str.capitalize())

app = Flask(__name__)

@app.route("/")
def ind():
    return render_template("index.html")


@app.route("/km")
def km():
    return render_template("kindex.html")




@app.route("/result",methods=['GET','POST'])
def result():
  # if(request.method=='POST'):
  #   print(request.form['mn1'])
  #   print(request.form['mn2'])
  # print(request.method)
  df=pd.read_csv("datasets/movie_dataset.csv")
  data = df.filter(['index','title','genres'], axis=1)
  features=['index','title','genres']
  for feature in features:
    data[feature] = data[feature].fillna('')
  x = data.genres
  a = list()
  for i in x:
    abc = i
    a.append(abc.split(' '))
  a = pd.DataFrame(a)   
  b = a[0].unique()
  for i in b:
    data[i] = 0
  for i in b:
    data.loc[data['genres'].str.contains(i), i] = 1
  data = data.drop(['index','genres','title'],axis =1)
  kmeanModel = KMeans(n_clusters=8)
  kmeanModel.fit(data)    
  data['Cluster'] = kmeanModel.labels_
  def get_title_from_index(index):
    return df[df['index']==index]["title"].values[0]
  def get_index_from_title(title):
      if len(df[df['title']==title]["index"].values)==0:
          return -1
      return df[df['title']==title]["index"].values[0]
  def get_director_from_index(index):
    return df[df['index']==index]["director"].values[0]
  def get_keywords_from_index(index):
    return df[df['index']==index]["keywords"].values[0]
  def get_cast_from_index(index):
    return df[df['index']==index]["cast"].values[0]

  movie_user_likes = request.form['movie']
  print(movie_user_likes)


  movie_index = get_index_from_title(movie_user_likes)
  # print(movie_index)
  if movie_index==-1: return render_template("404.html")

  d=[[0,movie_user_likes,get_director_from_index(movie_index),get_cast_from_index(movie_index),get_keywords_from_index(movie_index)]]
  new_df=pd.DataFrame(d, columns=['index','title','director','cast','keywords'])
  flag=data['Cluster'][movie_index]
  
  j=1
  for i in data['Cluster']:
    if i==flag:
      row={'index':j,'title':get_title_from_index(j), 'director':get_director_from_index(j), 'cast':get_cast_from_index(j), 'keywords':get_keywords_from_index(j)}
      new_df = new_df.append(row, ignore_index=True)  
      j=j+1
  features = ['keywords','cast','director']
  for feature in features:
    new_df[feature] = new_df[feature].fillna('') 

  def combine_features(row):
    return row['keywords']+" "+row['cast']+" "+row['director']
  
  new_df["combined_features"] = new_df.apply(combine_features,axis=1) 
  cv = CountVectorizer() 
  count_matrix = cv.fit_transform(new_df["combined_features"])
  
  cosine_sim = cosine_similarity(count_matrix)
  similar_movies = list(enumerate(cosine_sim[0])) 
  sorted_similar_movies = sorted(similar_movies,key=lambda x:x[1],reverse=True)[1:]
  def get_title_from_index2(index):
    return new_df[new_df['index']==index]["title"].values[0]
  def get_director_from_index2(index):
# print(index)
    return new_df[new_df['index']==index]["director"].values[0]
  def get_keywords_from_index2(index):
# print(index)
    return new_df[new_df['index']==index]["keywords"].values[0]
  def get_cast_from_index2(index):
# print(index)
    return new_df[new_df['index']==index]["cast"].values[0]
  def get_gen_from_index2(index):
# print(index)
    return df[df['index']==index]["genres"].values[0]
  def get_rat_from_index2(index):
# print(index)
    return df[df['index']==index]["vote_average"].values[0]
  i=0
  movie_name=[]
  movie_gen=[]
  movie_rat=[]
  #print("Top 5 similar movies to "+movie_user_likes+" are:\n")
  for element in sorted_similar_movies:
    #print(get_title_from_index2(element[0]))
    movie_name.append(get_title_from_index2(element[0]))
    movie_gen.append(get_gen_from_index2(element[0]))
    movie_rat.append(get_rat_from_index2(element[0]))
    i=i+1
    if i>6:
      break
  #print(movie_name)
  movie_name1=movie_name[1]
  movie_name2=movie_name[2]
  movie_name3=movie_name[3]
  movie_name4=movie_name[4]
  movie_name5=movie_name[5]
#   print(get_director_from_index(movie_index))
  movie_dir=get_director_from_index(movie_index)
  movie_cast=get_cast_from_index(movie_index)
  print(movie_index)
  print(df[df.index==movie_index].title)
  print(str(df[df.index==movie_index].overview.iloc[0]))
  overview=str(df[df.index==movie_index].overview.iloc[0])
  
  vote_rating=str(df[df.index==movie_index].vote_count.iloc[0])
  vote_avg=str(df[df.index==movie_index].vote_average.iloc[0])
  genre=str(df[df.index==movie_index].genres.iloc[0])
  release_date=str(df[df.index==movie_index].release_date.iloc[0])
  runtime=str(df[df.index==movie_index].runtime.iloc[0])
#   status="good"
#   if float(vote_avg) < 5 : status="bad"

#   print(movie_cast.split(" "))
  print(df.columns)
  movie_cast=movie_cast.split(" ")
  return render_template("result.html",movie_user_likes=movie_user_likes,movie_name1=movie_name1,movie_name2=movie_name2,
  movie_name3=movie_name3,movie_name4=movie_name4,movie_name5=movie_name5,
  movie_dir=movie_dir,movie_cast=movie_cast,overview=overview,vote_count=vote_rating,vote_avg=vote_avg,
  genres=genre,release_date=release_date,runtime=runtime,movie_key=movie_gen,movie_rat=movie_rat)
  


@app.route("/home")
def home():
    suggestions = get_suggestions()
    return render_template('home.html',suggestions=suggestions)

@app.route("/similarity",methods=["POST"])
def similarity():
    movie = request.form['name']
    rc = rcmd(movie)
    if type(rc)==type('string'):
        return rc
    else:
        m_str="---".join(rc)
        return m_str

@app.route("/recommend",methods=["POST"])
def recommend():
    # getting data from AJAX request
    title = request.form['title']
    cast_ids = request.form['cast_ids']
    cast_names = request.form['cast_names']
    cast_chars = request.form['cast_chars']
    cast_bdays = request.form['cast_bdays']
    cast_bios = request.form['cast_bios']
    cast_places = request.form['cast_places']
    cast_profiles = request.form['cast_profiles']
    imdb_id = request.form['imdb_id']
    poster = request.form['poster']
    genres = request.form['genres']
    overview = request.form['overview']
    vote_average = request.form['rating']
    vote_count = request.form['vote_count']
    release_date = request.form['release_date']
    runtime = request.form['runtime']
    status = request.form['status']
    rec_movies = request.form['rec_movies']
    rec_posters = request.form['rec_posters']

    # get movie suggestions for auto complete
    suggestions = get_suggestions()

    # call the convert_to_list function for every string that needs to be converted to list
    rec_movies = convert_to_list(rec_movies)
    rec_posters = convert_to_list(rec_posters)
    cast_names = convert_to_list(cast_names)
    cast_chars = convert_to_list(cast_chars)
    cast_profiles = convert_to_list(cast_profiles)
    cast_bdays = convert_to_list(cast_bdays)
    cast_bios = convert_to_list(cast_bios)
    cast_places = convert_to_list(cast_places)
    
    # convert string to list (eg. "[1,2,3]" to [1,2,3])
    cast_ids = cast_ids.split(',')
    cast_ids[0] = cast_ids[0].replace("[","")
    cast_ids[-1] = cast_ids[-1].replace("]","")
    
    # rendering the string to python string
    for i in range(len(cast_bios)):
        cast_bios[i] = cast_bios[i].replace(r'\n', '\n').replace(r'\"','\"')
    
    # combining multiple lists as a dictionary which can be passed to the html file so that it can be processed easily and the order of information will be preserved
    movie_cards = {rec_posters[i]: rec_movies[i] for i in range(len(rec_posters))}
    
    casts = {cast_names[i]:[cast_ids[i], cast_chars[i], cast_profiles[i]] for i in range(len(cast_profiles))}

    cast_details = {cast_names[i]:[cast_ids[i], cast_profiles[i], cast_bdays[i], cast_places[i], cast_bios[i]] for i in range(len(cast_places))}

    # web scraping to get user reviews from IMDB site
    sauce = urllib.request.urlopen('https://www.imdb.com/title/{}/reviews?ref_=tt_ov_rt'.format(imdb_id)).read()
    soup = bs.BeautifulSoup(sauce,'lxml')
    soup_result = soup.find_all("div",{"class":"text show-more__control"})

    reviews_list = [] # list of reviews
    reviews_status = [] # list of comments (good or bad)
    for reviews in soup_result:
        if reviews.string:
            reviews_list.append(reviews.string)
            # passing the review to our model
            movie_review_list = np.array([reviews.string])
            movie_vector = vectorizer.transform(movie_review_list)
            pred = clf.predict(movie_vector)
            reviews_status.append('Good' if pred else 'Bad')

    # combining reviews and comments into a dictionary
    movie_reviews = {reviews_list[i]: reviews_status[i] for i in range(len(reviews_list))}     
    # print(cast_details)
    # passing all the data to the html file
    return render_template('recommend.html',title=title,poster=poster,overview=overview,vote_average=vote_average,
        vote_count=vote_count,release_date=release_date,runtime=runtime,status=status,genres=genres,
        movie_cards=movie_cards,reviews=movie_reviews,casts=casts,cast_details=cast_details)

if __name__ == '__main__':
    app.run(port=8000,debug=True)
