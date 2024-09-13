This folder contains the emerging risk text mining models created for HOLiFOOD.

**_Input data for topic modelling_**

The Europe Media Monitor (EMM) is a sophisticated news aggregation tool developed by the European Commission's Joint Research Centre (JRC). It primarily focuses on monitoring and analyzing media sources to provide timely and relevant information on diverse topics, including food safety. EMM aggregates data from a wide array of sources, including online news websites, blogs, and other digital media outlets. It focuses on publicly available news content. The system uses advanced algorithms to filter and categorize the retrieved news articles, ensuring that only relevant content is included in the food safety dataset. This helps in reducing noise and focusing on pertinent information.  
The Europe Media Monitor (EMM) was utilized as a key source of input data for topic modeling in this project. EMM aggregates news articles from various sources, and for this study, the food safety category was selected to ensure relevance. Textual data is retrieved weekly through web scraping, allowing for the continual collection of current information. 

**_Selection of the most fit-for purpose topic detection method_**

Different topic detection tools have been selected to be tested/benchmarked. Data retrieval, text pre-processing and topic modeling workflows have been developed in KNIME software. Food safety news from Europe Media Monitor have been retrieved as input for topic modeling. A workflow, including data retrieval and text pre-processing algorithms for co-occurrence network analysis of EMM news have been developed. Based on algorithm testing with demo datasets for different topic modeling frameworks and visualization options, dynamic BERTopic has been selected as the best option. 

**_Building up the BERTopic framework _**
  
_Data retrieval_

The above-mentioned EMM food safety news served as input textual data. During the development, different approaches were used for finding the best for our purpose.  
- Data retrieval with KNIME software – at first, built-in nodes in KNIME were used to retrieve textual data 
- Data retrieval in Python – as BERTopic framework is implemented in Python, data retrieval was also implemented in Python in order to have a single workflow 
- Web-scraping: For a training database, text of different news was scraped from the web with the Beautiful Soup python library (Richardson, 2015). As EMM does not store news, however, news links were collected from recent years rigorously, in order to build an adequate size training database.

_Adjustment and fine-tuning of parameters in BERTopic framework_

We used an up-to-date language model (all-mpnet-base-v2) (Reimers, n.d.) for improved embedding performance for the first step.   
Then, most of the parameters were used as the framework offers by default, however, some adjustments were made in the dimensionality reduction and clustering algorithms. Parameters have been set to minimize the number of uncategorized documents (thus reducing noise) and to help the creation of meaningful topic clusters (Farea et al., 2024). This means (n_neighbors=14, n_components=5, min_dist=0.0, metric='cosine', random_state=42) for UMAP and (min_cluster_size=3, metric='euclidean', cluster_selection_method='eom', prediction_data=True) for HBDSCAN. 
Custom topic representation keyword model (KeyBERT) (Grootendorst, 2020) was also employed to enhance lucidity.  
