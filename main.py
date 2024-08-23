import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions
from sklearn.metrics import accuracy_score

st.title("Decision Tree")

def dtreeanalyzer(md,ct,mst):

    df=pd.read_csv("hyperplane.csv")

    x=df.iloc[:,:-1]
    y=df.iloc[:,-1]
    
    fig, ax = plt.subplots()

    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
    dt=DecisionTreeClassifier(max_depth=md,criterion=ct,min_samples_split=mst)
    dt.fit(x_train,y_train)
    
    y_pred=dt.predict(x_test)
    ac=accuracy_score(y_test,y_pred)
    
    plot_decision_regions(x_train.values,y_train.values,clf=dt,legend=2,ax=ax)
    st.pyplot(fig)
    
    st.write("Accuracy:",ac)
    
  
    plot_tree(dt)
    st.pyplot(fig)

 
crt=['gini','entropy','log_loss']

mxd=st.sidebar.slider("Select Depth Of Tree",min_value=1,max_value=20)
mst=st.sidebar.slider("Minimun No.of Samples for Splitting a Node",min_value=2)
ct=st.sidebar.selectbox("Select Criterion",options=crt)
   
if st.button("Run"):
    dtreeanalyzer(mxd,ct,mst)
    


    

    

