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
st.title('Prediksi Batu Ginjal Berdasarkan Analisis Urine ')
st.write("""
#### Dengan 4 Pilihan Metode Klasifikasi
###### Mana yang terbaik?
""")
st.write("""
###### Dataset diambil dari halaman https://www.kaggle.com/datasets/vuppalaadithyasairam/kidney-stone-prediction-based-on-urine-analysis
###### Untuk Tipe Datanya sendiri berupa tipe data Numerik
###### Data tersebut berisi hasil Analisis dari tes Urine dimana terdapat atribut atau fitur seperti gravity, ph, osmo, cond, urea, calc, dan terdapat target yang berisi hasil klasifikasinya 
""")
# Dataset
dataset = pd.read_excel(r"data.xlsx")