# %%writefile recommender.py
import streamlit as st
import pickle
import numpy as np
import pandas as pd
import json
from sympy import expand
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.colors as colors


st.set_page_config(
    page_title="User Recommendations",
    layout="wide"
)

@st.cache_data()
def persistlistens():
    return {}

def sparse_list(ratings, artist_ids):
  output = [0] * len(artist_ids)
  not_found = []
  for k, v in ratings.items():
    try:
      output[k] = v
    except ValueError:
      not_found.append(k)

  return output, not_found

@st.cache_data()
def retrieve_item_dict():
  with open('drive/MyDrive/CA4015/artists.pickle', 'rb') as p:
    item_dict = pickle.load(p)
  return item_dict

@st.cache_data()
def retrieve_artist_embeddings():
  with open('drive/MyDrive/CA4015/embeddings.pickle', 'rb') as p:
    embeddings = pickle.load(p)
  return embeddings

with open('drive/MyDrive/CA4015/trained_model.pickle', 'rb') as p:
  trained_model = pickle.load(p)

# @st.cache_data()
# def retrieve_ratings_table():
#   with open('/content/drive/MyDrive/CA4015/dataset/processed/ratings_table.csv', 'r') as file:
#     ratings_table = pd.read_csv(file)
#   return ratings_table

@st.cache_data()
def compute_recommendations(ratings, artist_id_name):
  embeddings = retrieve_artist_embeddings()
  
  rating_vector, not_found = sparse_list(ratings, artist_id_name.keys())
  ratings = np.asarray(rating_vector)

  latent_rep = np.matmul(ratings, embeddings)
  recommendations = np.matmul(latent_rep, np.transpose(embeddings))
  names = [artist_id_name[artist_id] for artist_id in artist_id_name.keys()]

  return zip(names, recommendations), not_found

@st.cache_data()
def compute_scores(query_embedding, item_embeddings, measure="dot"):
    """Computes the scores of the candidates given a query.
    Args:
      query_embedding: a vector of shape [k], representing the query embedding.
      item_embeddings: a matrix of shape [N, k], such that row i is the embedding
        of item i.
      measure: a string specifying the similarity measure to be used. Can be
        either DOT or COSINE.
    Returns:
      scores: a vector of shape [N], such that scores[i] is the score of item i.
    """
    u = query_embedding
    V = item_embeddings
    if measure == "cosine":
        V = V / np.linalg.norm(V, axis=1, keepdims=True)
        u = u / np.linalg.norm(u)
    scores = u.dot(V.T)
    return scores

@st.cache_data()
def artist_neighbors(model, artists, substring, measure, k=6):
    embeddings = retrieve_artist_embeddings()
    ids = artists[artists['name'].str.contains(substring)].index.values
    titles = artists.iloc[ids]['name'].values
    if len(titles) == 0:
        raise ValueError("Found no artists with the name %s" % substring)
    print("Nearest neighbors of: %s." % titles[0])
    if len(titles) > 1:
        print("[Found more than one matching artist. Other candidates: {}]".format(
            ", ".join(titles[1:])))
    artistID = ids[0]
    scores = compute_scores(
        embeddings[artistID], embeddings,
        measure)
    score_key = measure + 'Score'
    df = pd.DataFrame({
        score_key: list(scores),
        'name': artists['name']
    })
    return df.sort_values([score_key], ascending=False).head(k)

def main():
# For plotting purposes
  template = dict(
  layout = go.Layout(
    title = dict(
      x = 0.5,
      xanchor = "center"
      )
    )
  )
  # Generate a gradual color set
  color_scale = colors.sequential.Agsunset


  st.markdown('---')
  st.header("User Recommendations")
  st.markdown('---')

  artist_id_name = retrieve_item_dict()
  artist_name_id = {v: k for k, v in artist_id_name.items()}
  artist_names = artist_id_name.values()

  user_listens = persistlistens()

  if "recommendations" not in locals():
    recommendations = None

  left, right = st.columns([2, 5])

  left.subheader("User input")
  right.subheader("Your recommendations can be seen here once submitted")
  selected_artists = left.multiselect("Select artist to rate", artist_names)
  expander =  left.expander("Rate your selected artists", expanded=False)

  user_listens.clear()
  for artist in selected_artists:

    rating = expander.selectbox(f"Rating for {artist}", [0,1,2,3,4,5])

    id = artist_name_id[artist]
    user_listens[id] = rating
  
  if len(user_listens) > 0:
    output, not_found = compute_recommendations(user_listens, artist_id_name)
    # right.dataframe(pd.DataFrame(output, columns=["Items", "Predicted relevance"]).sort_values(by=["Predicted relevance"], ascending=False).reset_index(drop=True, inplace=True).index += 1)
    result_df = pd.DataFrame(output, columns=["Items", "Predicted relevance"]).sort_values(by=["Predicted relevance"], ascending=False)
    result_df.reset_index(drop=True, inplace=True)
    result_df.index += 1
    right.dataframe(result_df)

    
  # Load artist information DataFrame
  artists_info = pd.read_csv('/content/drive/MyDrive/CA4015/dataset/processed/artists_info.csv')
  
  st.markdown('---')
  st.header('Cosine/Dot Recommendation')
  st.markdown('---')
  st.subheader('User Input')
  c1,c2,c3 = st.columns([2,2,3])
  selected_artist = c1.selectbox('Artist Name', artist_names)
  # selected_artist = expander.selectbox('Artist Name', artist_names)
  # user_listens.clear()
  col1, col2, col3 = st.columns([2,2,3])

  

  rec_cos = artist_neighbors(trained_model, artists_info, selected_artist, measure='cosine', k=6)
  col1.markdown('---')
  col1.subheader('Cosine Recommendation Results')
  col1.dataframe(rec_cos)



  fig_regcos = go.Figure(
      data = go.Bar(dict(
          x = rec_cos['name'],
          y = rec_cos['cosineScore'],
          marker=dict(
              color = color_scale
          ),
      )),
      layout = dict(
          template = template,
          xaxis = dict(title = 'Recommended Artists', zeroline=True),
          yaxis = dict(title = 'Cosine Score', zeroline=True),
      )
  )
  col2.plotly_chart(fig_regcos)

  left_col, mid_col, right_col = st.columns([2,2,3])

  rec_dot = artist_neighbors(trained_model, artists_info, selected_artist, measure='dot', k=6)
  left_col.markdown('---')
  left_col.subheader('Dot Recommendation Results')
  left_col.dataframe(rec_dot)

  fig_regdot = go.Figure(
    data = go.Bar(dict(
        x = rec_dot['name'],
        y = rec_dot['dotScore'],
        marker=dict(
            color = color_scale
        ),
    )),
    layout = dict(
        template = template,
        xaxis = dict(title = 'Recommended Artists', zeroline=True),
        yaxis = dict(title = 'Dot Score', zeroline=True),
      )
  )
  mid_col.plotly_chart(fig_regdot)
  st.markdown('---')

  # Set header and subheader
  st.header("Query for Artist Information")
  st.markdown('---')

  cl1,cl2,cl3 = st.columns([2,2,3])
  cl1.subheader("Search for an Artist")

  left_column, right_column = st.columns([2, 5])

  # Get user input for artist name
  # search_artist_val_name = left_column.text_input("Enter artist name:")
  search_artist_val_name = left_column.selectbox("Select artist to query", artist_names)

  # Query artist information based on artist name
  search_artist_name = artists_info.query("name == @search_artist_val_name")

  # Display artist information if found, or show a message if not found
  if not search_artist_name.empty:
      search_artist_index = search_artist_name.index[0]
      search_artist_spotifyurl = search_artist_name.loc[search_artist_index]["spotifyUrl"]
      search_artist_img = search_artist_name.loc[search_artist_index]["imageUrl"]
    
      # Display artist name and Spotify URL
      cl2.subheader("Here is your search result")
      right_column.markdown(f"**Artist Name:** {search_artist_val_name}")
      right_column.markdown(f"**Spotify URL:** [{search_artist_spotifyurl}]({search_artist_spotifyurl})")

      # Display artist image
      right_column.image(search_artist_img, caption=search_artist_val_name, width=500)
      
      
      # # Display artist name and Spotify URL
      # st.markdown(f"**Artist Name:** {search_artist_val_name}")
      # st.markdown(f"**Spotify URL:** [{search_artist_spotifyurl}]({search_artist_spotifyurl})")

      # # Display artist image
      # st.image(search_artist_img, caption=search_artist_val_name, width=400, clamp=False)
  else:
      st.warning("No results found for the artist name: " + search_artist_val_name)
  st.markdown('---')



  # filtered_ratings = user_ratings[user_ratings["implicitRating"] == 4]

  # Display the filtered DataFrame
  # st.dataframe(filtered_ratings)
  # artists = pd.read_csv("drive/MyDrive/CA4015/dataset/raw/artists.dat", sep='\t', encoding='latin-1')
  # artists = pd.read_csv()
  # st.dataframe(artists)
  # cells = notebook['cells']
  # dataframes = [cell['outputs'][0]['data']['text/plain'] for cell in cells if cell['cell_type'] == 'code' and cell['outputs'] and 'data' in cell['outputs'][0] and 'text/plain' in cell['outputs'][0]['data'] and 'ratings_table' in cell['source']]
  # ratings_table = pd.read_csv(dataframes[0], delimiter='\t')  # Adjust delimiter if needed

  # Display the DataFrame

if __name__ == '__main__':
  main()
