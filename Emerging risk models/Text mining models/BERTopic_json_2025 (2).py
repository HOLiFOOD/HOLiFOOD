import pandas as pd
import json
import time
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
import plotly.express as px
from datetime import datetime

print("CUDA available:", torch.cuda.is_available())
start_time = time.time()

# -----------------------------
# Load Data from JSON
# -----------------------------
input_json = "data.json"  # Update with your JSON file path
with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)
df = pd.DataFrame(data)
print("Data preview:")
print(df.head())

# Extract documents, dates, and URLs.
documents = df["Summary"].to_list()
dates = pd.to_datetime(df["Scrape Date"]).tolist()

# -----------------------------
# Topic Modeling Pipeline Setup
# -----------------------------
embedding_model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")
embeddings = embedding_model.encode(documents, show_progress_bar=True)

umap_model = UMAP(
    n_neighbors=15,
    n_components=5,
    min_dist=0.0,
    metric="cosine",
    random_state=42
)
hdbscan_model = HDBSCAN(
    min_cluster_size=3,
    metric="euclidean",
    cluster_selection_method="eom",
    prediction_data=True
)
vectorizer_model = CountVectorizer(
    stop_words="english",
    min_df=2,
    ngram_range=(1, 2)
)

# Representation model
keybert_model = KeyBERTInspired()
representation_model = {"KeyBERT": keybert_model}

topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    representation_model=representation_model,
    top_n_words=10,
    verbose=True
)

# -----------------------------
# Train Topic Model
# -----------------------------
topics, probs = topic_model.fit_transform(documents, embeddings)
topic_info = topic_model.get_topic_info()
print("Topic info preview:")
print(topic_info.head())

# -----------------------------
# Augment original DataFrame with topic assignments
# -----------------------------
df["Assigned_Topic"] = topics
df["Topic_Probability"] = [
    float(p.max()) if isinstance(p, np.ndarray) else float(p)
    for p in probs
]

def topic_keywords(topic_num, top_n=5):
    if topic_num == -1:
        return []
    kws = topic_model.get_topic(topic_num)
    return [word for word, _ in kws[:top_n]]

df["Topic_Keywords"] = df["Assigned_Topic"].apply(lambda t: topic_keywords(t, top_n=5))

# -----------------------------
# Save augmented data back to JSON
# -----------------------------
output_json = "input_with_topics.json"
records = df.to_dict(orient="records")
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(records, f, ensure_ascii=False, indent=2)
print(f"Augmented data with topic assignments saved to {output_json}")

# -----------------------------
# Create Custom Labels for Pie Chart
# -----------------------------
def get_topic_label(topic_num):
    if topic_num == -1:
        return "Outlier"
    kws = topic_model.get_topic(topic_num)
    top_keywords = ", ".join([word for word, _ in kws[:3]])
    return f"Topic {topic_num}: {top_keywords}"

topic_info["Label"] = topic_info["Topic"].apply(get_topic_label)

# -----------------------------
# Compute Representative Documents
# -----------------------------
df_topics = pd.DataFrame({
    "Document": documents,
    "Topic": topics,
    "URL": df["URL"].tolist()
})
rep_docs = {}
for topic_num in sorted(df_topics["Topic"].unique()):
    if topic_num == -1:
        continue
    docs = (
        df_topics[df_topics["Topic"] == topic_num]
        [["Document", "URL"]]
        .drop_duplicates()
    )
    rep_docs[topic_num] = docs.head(50).to_dict(orient="records")

def rep_docs_to_string(topic_num):
    if topic_num in rep_docs:
        return "; ".join(
            f"{d['Document'][:50].replace(chr(10), ' ')} ({d['URL']})"
            for d in rep_docs[topic_num]
        )
    return "None"

filtered_info = topic_info[topic_info.Topic != -1].copy()
filtered_info["RepDocs"] = filtered_info["Topic"].apply(rep_docs_to_string)

# -----------------------------
# Visualizations
# -----------------------------
# 1. Intertopic Distance Map
fig_topics = topic_model.visualize_topics()
fig_topics.write_html("intertopic_distance.html")

# 2. Bar Chart of Top Topics
fig_barchart = topic_model.visualize_barchart(top_n_topics=20)
fig_barchart.write_html("topic_barchart.html")

# 3. Pie Chart with Custom Labels and Representative Documents
filtered_info["TopicID"] = filtered_info["Topic"]
fig_pie = px.pie(
    filtered_info,
    names="Label",
    values="Count",
    title="Topic Distribution (Pie Chart)",
    hover_data=["RepDocs", "TopicID"]
)
fig_pie.update_traces(
    hovertemplate='%{label}<br>Count: %{value}<extra></extra>',
    customdata=filtered_info["TopicID"].tolist()
)

# Save the pie chart HTML
html_filename = "topic_distribution_pie.html"
fig_pie.write_html(html_filename, include_plotlyjs="cdn")

# -----------------------------
# Append Original JavaScript Snippet for Interactive Click Handling
# -----------------------------
# Serialize the candidate_docs dictionary to JSON.
candidate_docs = {int(k): v for k, v in rep_docs.items()}
candidate_docs_js = json.dumps(candidate_docs)

custom_script = f"""
<script>
  // Candidate documents for each topic, available for sampling.
  var candidateDocs = {candidate_docs_js};

  // Find the Plotly chart div.
  var myPlot = document.getElementsByClassName('plotly-graph-div')[0];

  // Utility function: Randomly sample n items from an array.
  function getRandomSamples(arr, n) {{
    var result = [];
    var taken = [];
    n = Math.min(n, arr.length);
    while (result.length < n) {{
      var index = Math.floor(Math.random() * arr.length);
      if (!taken.includes(index)) {{
        taken.push(index);
        result.push(arr[index]);
      }}
    }}
    return result;
  }}

  // Listen for click events on the pie chart.
  myPlot.on('plotly_click', function(data) {{
    var topic_id = data.points[0].customdata;
    if (Array.isArray(topic_id)) {{
      topic_id = topic_id[0];
    }}
    var docs = candidateDocs[topic_id];
    var maxToShow = 5;
    var sampledDocs = docs.length <= maxToShow ? docs : getRandomSamples(docs, maxToShow);

    // Create or select a div to display the representative documents.
    var displayDiv = document.getElementById('docDisplay');
    if (!displayDiv) {{
      displayDiv = document.createElement('div');
      displayDiv.id = 'docDisplay';
      displayDiv.style.marginTop = '20px';
      displayDiv.style.border = '1px solid black';
      displayDiv.style.padding = '10px';
      document.body.appendChild(displayDiv);
    }}

    // Build HTML: list each doc with a snippet and a clickable URL.
    var html = '<h3>Representative Documents for Topic ' + topic_id + '</h3><ul>';
    sampledDocs.forEach(function(doc) {{
      var snippet = doc.Document.substring(0, 50).replace(/\\n/g, ' ');
      html += '<li>' + snippet + ' (<a href="' + doc.URL + '" target="_blank">Link</a>)</li>';
    }});
    html += '</ul>';
    displayDiv.innerHTML = html;
  }});
</script>
"""

with open(html_filename, "r+", encoding="utf-8") as f:
    html_content = f.read()
    html_content = html_content.replace("</body>", custom_script + "\n</body>")
    f.seek(0)
    f.write(html_content)
    f.truncate()

print("Interactive pie chart saved as:", html_filename)

# 4. Total Distribution Bar Chart
fig_total = px.bar(
    topic_info,
    x="Topic",
    y="Count",
    title="Total Topic Distribution",
    hover_data=["Name"]
)
fig_total.write_html("total_topic_distribution_bar.html")

# 5. Topics Over Time
topics_over_time = topic_model.topics_over_time(
    documents,
    dates,
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=40
)
fig_topics_over_time = topic_model.visualize_topics_over_time(
    topics_over_time,
    top_n_topics=15
)
fig_topics_over_time.write_html("topics_over_time.html")

# 6. Hierarchy
fig_hierarchy = topic_model.visualize_hierarchy(top_n_topics=20)
fig_hierarchy.write_html("topic_hierarchy.html")

elapsed_time = time.time() - start_time
print(f"Topic Modeling and Visualization took {elapsed_time:.2f} seconds.")

