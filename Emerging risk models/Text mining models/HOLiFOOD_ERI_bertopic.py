import pandas as pd
from bertopic import BERTopic
import time
import torch
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.representation import KeyBERTInspired
from keybert import KeyBERT
import plotly.io as pio

print(torch.cuda.is_available())
start_time = time.time()

# Read csv file
df = pd.read_csv('cleaned_data2.csv')
print(df.head())

urls = df['url'].to_list()
titles = df['content'].tolist()
dates = df['date'].apply(lambda x: pd.Timestamp(x)).to_list()

# Ensure all titles are strings and there are no NaN values
titles = [str(title) for title in titles if isinstance(title, str) and not pd.isna(title)]

# Pre-calculate embeddings with the highest performing sentence transformer
embedding_model = SentenceTransformer("all-mpnet-base-v2")
embeddings = embedding_model.encode(titles, show_progress_bar=True)

# Dimensionality reduction with modified parameters
umap_model = UMAP(n_neighbors=14, n_components=5, min_dist=0.0, metric='cosine', random_state=42)

# Clustering with reduced clustersize
hdbscan_model = HDBSCAN(min_cluster_size=3, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# Tokenizer
vectorizer_model = CountVectorizer(stop_words="english", min_df=2, ngram_range=(1, 2))

# Representations
# KeyBERT
keybert_model = KeyBERTInspired()

# All representation models
representation_model = {
    "KeyBERT": keybert_model,
}

topic_model = BERTopic(
  # Pipeline models
  embedding_model=embedding_model,
  umap_model=umap_model,
  hdbscan_model=hdbscan_model,
  vectorizer_model=vectorizer_model,
  representation_model=representation_model,

  # Hyperparameters
  top_n_words=10,
  verbose=True
)

# Train model
topics, probs = topic_model.fit_transform(titles, embeddings)

# Identified topics descriptives
freq = topic_model.get_topic_info()
print("Number of topics: {}".format(len(freq)))
print(freq.head())

# Show topics
topic_model.get_topic_info()

# Visualize intertopic distance
topic_model.visualize_topics().show()   

# Visualize Topics using Bar Chart
topic_model.visualize_barchart(top_n_topics=40).show()

# Topics Over Time
topics_over_time = topic_model.topics_over_time(titles, dates, 
                                                global_tuning=True, evolution_tuning=True, nr_bins=40)

topic_model.visualize_topics_over_time(topics_over_time, topics=[0,1,2,3,4,5,7,9,10,11,12,13,14,16,20,22,23,24,25,26,27,28]).show()

# Visualize connections between topics using hierarchical clustering
topic_model.visualize_hierarchy(top_n_topics=15).show()

# Visualize Heatmap
topic_model.visualize_heatmap(n_clusters=30, width=1000, height=1000).show()

# Visualize Documents with plotly
#topic_model.visualize_documents(titles, embeddings=embeddings)

# Select most 5 similar topics
similar_topics, similarity = topic_model.find_topics("food security", top_n=5)
print(similar_topics)
most_similar = similar_topics[0]
print("Most Similar Topic Info: \n{}".format(topic_model.get_topic(most_similar)))
print("Similarity Score: {}".format(similarity[0]))

# Add topics, URLs, and dates to the DataFrame
df = pd.DataFrame({"Document": titles, "Topic": topics, "url": urls, "Date": dates})

elapsed_time = time.time() - start_time
print(f"Topic Modeling took {elapsed_time:.2f} seconds.")

topics, _ = topic_model.fit_transform(titles, embeddings)
df = pd.DataFrame({"Document": titles, "Topic": topics, "Date": dates})
print(df)

topic_number = 3
topic_model.get_topic_info(topic_number)
documents_from_topic = [doc for doc, topic in zip(titles, topics) if topic == topic_number]

# Print documents or process them further
print("Documents from Topic #10:")
for doc in documents_from_topic:
    print(doc)

kw_model = KeyBERT()  # Now KeyBERT is defined

# Get the keyphrases for each document
keyphrases = [kw_model.extract_keywords(doc) for doc in titles]
keyphrases = ["; ".join([kw[0] for kw in kp]) for kp in keyphrases]  # Convert list of tuples to string

# Add the topics, keyphrases, URLs, and dates to the DataFrame
df['Topic_ID'] = topics
df['KeyBERT_Keywords'] = keyphrases
df['url'] = urls
df['Date'] = dates

# Print column names to verify
print(df.columns)

# Save the updated DataFrame to the original CSV file
updated_data_file_path = 'eriscrape_withtopics.csv'
df.to_csv(updated_data_file_path, index=False)
print(f"Updated data saved to {updated_data_file_path}")

# Save documents with their categories to a new file
documents_with_categories_file_path = 'eriscrape_withtopicdefinitions3.csv'
df[['Document', 'Topic_ID', 'KeyBERT_Keywords', 'Date']].to_csv(documents_with_categories_file_path, index=False)
print(f"Documents with categories saved to {documents_with_categories_file_path}")

reduced_embeddings = UMAP(n_neighbors=10, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
topic_model.visualize_document_datamap(titles, embeddings=embeddings)

fig_topics = topic_model.visualize_topics()
pio.write_html(fig_topics, file="topics_visualization.html", auto_open=True)

# Visualize barchart
fig_barchart = topic_model.visualize_barchart()
pio.write_html(fig_barchart, file="barchart_visualization.html", auto_open=True)

# Visualize heatmap
fig_heatmap = topic_model.visualize_heatmap()
pio.write_html(fig_heatmap, file="heatmap_visualization.html", auto_open=True)

# Visualize hierarchy
fig_hierarchy = topic_model.visualize_hierarchy()
pio.write_html(fig_hierarchy, file="hierarchy_visualization.html", auto_open=True)

# Visualize topics over time
fig_dynamictopics = topic_model.visualize_topics_over_time(topics_over_time, topics=[0,1,2,3,4,5,7,9,10,11,12,13,14,16,20,22,23,24,25,26,27,28])
pio.write_html(fig_dynamictopics, file="dynamictopics.html", auto_open=True)
