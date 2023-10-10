import random

import gensim
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pyLDAvis.gensim_models
import regex
import seaborn as sns
import streamlit as st
from gensim import corpora
from gensim.models import CoherenceModel
from gensim.utils import simple_preprocess
from nltk.corpus import stopwords
import streamlit.components.v1 as components
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
from wordcloud import WordCloud
import matplotlib.colors as mcolors
import plotly.express as px

#Title
st.title('Topic Modelling ')
st.write("""
#### Dengan LDA Modelling
""")
st.write("""
###### Dataset diambil dari halaman https://pta.trunojoyo.ac.id/c_search/byprod/14
###### Untuk Tipe Datanya sendiri berupa tipe data Numerik
###### Data tersebut berisi data Abstrak yang telah di proses sehingga data yang dimunculkan berupa data vectorisasi  
""")
# Dataset
dataset = pd.read_excel(r"Data.xlsx")

st.write('')
st.write('## Dataset')
st.dataframe(data=dataset)
