import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
#import streamlit.components.v1 as components

# Web App Title
st.title('EDA Web App')
st.markdown("""
This app works with numpy, pandas, matplotlib and seaborn libraries to carry out simple exploratory data analysis on your data. \n
Authored by: Josiah Oborekanhwo \n
[Github](https://github/Josiah-Jovido)
""")

# Upload CSV data
with st.sidebar.header('1. Upload CSV file'):
    uploaded_file = st.sidebar.file_uploader('Upload your input CSV file')
    st.sidebar.markdown('''[Example CSV input file](https://github.com/josiah-jovido)''')

# Pandas Profiling report
if uploaded_file is not None:
    @st.cache
    def load_csv():
        csv_file = pd.read_csv(uploaded_file)
        return csv_file
    df = load_csv()
    d_type = df.dtypes
    des = df.describe()
    st.header('**Input DataFrame**')
    st.write(df)
    st.write('_ _ _')
    st.header('**Variable Types**')
    st.write(d_type)
    st.write('_ _ _')
    st.header('**Summary Statistics**')
    st.write(des)
    st.write('_ _ _')
    st.header('**Scatter plot**')
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], 5)
    st.pyplot()
    st.write('_ _ _')
    st.header('**Line Chart**')
    st.line_chart(df)
    st.header('**Correlation Heat Map**')
    st.set_option('deprecation.showPyplotGlobalUse', False)
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
else:
    st.info('Awaiting CSV file to be uploaded')
    if st.button('Press to use Example Dataset'):
        @st.cache
        def load_data():
            data = pd.DataFrame(np.random.rand(100, 5), columns=['temp','pressure','density','volume','weight'])
            return data
        df =  load_data()
        information = df.dtypes
        des = df.describe()
        st.header('**Input DataFrame**')
        st.write(df)
        st.write('_ _ _')
        st.header('**Variable Types**')
        st.write(information)
        st.write('_ _ _')
        st.header('**Summary Statistics**')
        st.write(des)
        st.write('_ _ _')
        st.header('**Scatter plot**')
        plt.scatter(df.iloc[:, 0], df.iloc[:, 1], 5)
        st.pyplot()
        st.write('_ _ _')
        st.header('**Line Chart**')
        st.line_chart(df)
        st.header('**Correlation Heat Map**')
        st.set_option('deprecation.showPyplotGlobalUse', False)
        corr = df.corr()
        mask = np.zeros_like(corr)
        mask[np.triu_indices_from(mask)] = True
        with sns.axes_style("white"):
            f, ax = plt.subplots(figsize=(7, 5))
            ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
        st.pyplot()
