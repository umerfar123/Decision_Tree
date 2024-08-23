import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

st.title("Decision Tree")

def dtreeanalyzer(md):

    df=pd.read_csv("hyperplane.csv")

    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    
    fig, ax = plt.subplots()

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    dt=DecisionTreeClassifier(max_depth=md)
    dt.fit(x_train,y_train)
    
    y_pred=dt.predict(x_test)
    ac=accuracy_score(y_test,y_pred)
    
    plot_decision_regions(x_train.values,y_train.values,clf=dt,legend=2,ax=ax)
    st.pyplot(fig)
    
    st.write("Accuracy:",ac)
    
  
    plot_tree(dt)
    st.pyplot(fig)

 

mxd=st.sidebar.slider("Select Depth Of Tree",min_value=1,max_value=20)
   
if st.button("Run"):
    dtreeanalyzer(mxd)
    


    

    

