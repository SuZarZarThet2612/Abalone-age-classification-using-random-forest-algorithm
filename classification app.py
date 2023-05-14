import streamlit as st
from streamlit_option_menu import option_menu
import streamlit.components.v1 as html
from  PIL import Image
import numpy as np
import io
import pandas as pd
import requests
from streamlit_lottie import st_lottie
import math
import os
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
import pickle
import plotly.graph_objects as go
import numpy as np
model = pickle.load(open('randomForestOrg2.pkl','rb'))

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Prediction",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded",
)
with st.sidebar:
    choose = option_menu("App Gallery", ["Home Page", "Information", "Classification", "Accurancy"],
                         icons=['house', 'info-circle', 'box', 'bar-chart-line-fill'],
                         menu_icon="app-indicator", default_index=0,
                         styles={
        "container": {"padding": "5!important", "background-color": "#fafafa"},
        "icon": {"color": "orange", "font-size": "25px"},
        "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "#02ab21"},
    }
    )


if choose == "Home Page":


        with st.container():
            image_column,text_column = st.columns((1,5))
            with image_column:
                    image = Image.open('utycclogo.jpg')
                    st.image(image,width=125)

            with text_column:
                    html_temp = """
                    <div style="background:white ;padding:1px">
                    <h4 style="color:#003399;text-align:center;">University of Technology (Yantanarpon Cyber City)</h4>
                    <h5 style="color:#003399;text-align:center;">Department of Information Science</h5>
                    </div>
                    """
                    st.markdown(html_temp, unsafe_allow_html = True)

        image = Image.open('96.png')
        html_temp = """
                    <div style="background:#02ab21 ;padding:10px">
                    <h2 style="color:white;text-align:center;">Classification of Abalone Age using Random Forest</h2>
                    </div>
                    """
        st.markdown(html_temp, unsafe_allow_html = True)

        st.image(image,
            use_column_width=True)
        with st.container():
            image_column,text_column = st.columns((1,0.3))
            with image_column:
                    st.write("Supervised by")
                    st.write("Dr. Naw Thiri Wai Khin")

            with text_column:
                   st.write("Presented by")
                   st.write("Ma Su Zar Zar Thet")
                   st.write("6IST-13R")



elif choose == "Information":

        def load_lottieurl(url):
            r=requests.get(url)
            if r.status_code!=200:
                return None
            return r.json()
        def load_lottieurl1(url):
            r=requests.get(url)
            if r.status_code!=200:
                return None
            return r.json()

        lottie_coding=load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_mf5j5kua.json")
        lottie_coding1=load_lottieurl1("https://assets2.lottiefiles.com/packages/lf20_mf5j5kua.json")
        abalonepage=Image.open("abalonehome.jpg")
        with st.container():
            st.subheader("Welcome from My Page:wave:")
            st.title("Information for Abalone")

            st.write("Abalone is a nutritious food resource in the many parts of the world and is considered as a luxury item.")
            st.write("The economic value of abalone is positively correlated with its age.")
            st.markdown("[Learn More>](#what-is-abalone)")


        with st.container():
            st.write("----")
            left_column,right_column = st.columns(2)
            with left_column:
                st.header("What my system do")
                st.write("Performs a classification to know abalone age.")
                st.write("Indicate the age of abalone (young, adult, old)")
                st.write(
                        """ 
                        On my system I am creating :
                        - Length : The longest measurement of the abalone shell in mm.
                        - Height : Height of the shell in mm.
                        - Whole Weight : Weight of the abalone in grams.
                        - Shell Weight : Weight of the abalone after being dried in grams.
                        
                        All Continuous numeric value.
                            """)

            with right_column:
                    st_lottie(lottie_coding,height=400,key="coding")
        st.write("----")
        st.header('What is Abalone?')
        with st.container():

            left_column,right_column = st.columns((1,2))
            with left_column:
                st.image(abalonepage)

            with right_column:
                st.write(
                        """ 
                       Abalone is nutritious food resource and farming in many parts of the world. 
                       They are types of the single-shelled marine snails found in the cold coastal waters worldwide, found along the coastal regions of the countries such as Australia, Western North America,South Africa,New Zealand, and Japan. 
                       Abalone pearl jewellery is majorly very popular in New Zealand and Australia, in no minor part due to the marketing and farming efforts of pearl companies. 
                            """)

        st.write(
                        """ 
                       The inner shell of the abalone is an iridescent swirl of intense colors, ranging from a deep cobalt blue and peacock green to purples, creams and pinks. 
                       Therefore, each pearl, natural or cultured, will have its own unique collage of colors. 
                       Abalone is considered a luxury item, and is traditionally reserved for the special occasions such as weddings and other celebrations. 
                       The economic value of abalone is correlated with its age. 
                       Therefore, to detect the age of an abalone accurately is important for both farmers and customers to determine its price.  
                       The age of the abalone is highly correlated to its prices as it is the sole factor used to determine it’s worth. 
                       Determining the age of an abalone is a  highly involved process  that is usually carried out in the laboratory. 
                       Technically, rings are formed in the inner shell of the abalone as it grows gradually at a rate of one ring per year. 
                       To get access to the inner rings of the abalones, the shell's outer rings need to be cut. 
                       After polishing and staining, a lab technician examines a shell sample under a microscope and counts the rings. 
                       Knowing the correct price of the abalone is important to both the farmers and consumers. 
                       In addition, knowing the correct age is mainly crucial to environmentalists who seek to protect this endangered species. 
                       Due to the inherent inaccuracy in the manual method of counting the rings and thus calculating the age, researchers have tried to employ the physical characteristics of an abalone such as sex, weight, height, and length to determine its age. 
                            """)

elif choose == "Classification":

    def predict_age(Length,Height,Whole_weight,Shell_weight,i,m,f):
            input=np.array([[Length,Height,Whole_weight,Shell_weight,i,m,f]]).astype(np.float64)
            prediction = model.predict(input)
            #pred = '{0:.{1}f}'.format(prediction[0][0], 2)
            return int(prediction)

    def main():


            html_temp = """
            <div style="background:#02ab21 ;padding:10px">
            <h2 style="color:white;text-align:center;">Abalone Age Classification</h2>
            </div>
            """
            st.markdown(html_temp, unsafe_allow_html = True)
            placeholder = st.empty()
            with st.form(key='myform'):

                Length = st.number_input("Length",min_value=0.000001,max_value=2.815000,format="%.6f")
                Height = st.number_input("Height",min_value=0.000001,max_value=3.130000,format="%.6f")
                Whole_weight = st.number_input("Whole weight",min_value=0.000001,max_value=4.825500,format="%.6f")
                Shell_weight= st.number_input("Shell_weight",min_value=0.000001,max_value=3.005000,format="%.6f")
                Infant = st.radio("Infant or not", ("Infant","Not Infant"))
                Male = st.radio("Male or not", ("Male","Not Male"))
                Female= st.radio("Female or not", ("Female","Not Female"))
                if Infant == "Infant":
                    i = 1
                else:
                    i = 0
                if Male == "Male":
                    m = 1
                else:
                    m = 0
                if Female == "Female":
                    f = 1
                else:
                    f = 0
                left_column,right_column = st.columns(2)
                with left_column:
                    submit_button=st.form_submit_button("Let's find out the age")
                    safe_html ="""  
                      <div style="background-color:#80ff80; padding:10px >
                      <h2 style="color:white;text-align:center;"> The Abalone is Young</h2>
                      </div>
                    """
                    warn_html ="""  
                      <div style="background-color:#F4D03F; padding:10px >
                      <h2 style="color:white;text-align:center;"> The Abalone is Adult</h2>
                      </div>
                    """
                    danger_html="""  
                      <div style="background-color:#F08080; padding:10px >
                       <h2 style="color:black ;text-align:center;"> The Abalone is Old</h2>
                       </div>
                    """


                    if submit_button:
                            if (i==1 & m==1 & f==1) :
                                st.markdown(":x: **:red[Please Choose Gender! ]** :x:")
                            else:
                                output = predict_age(Length,Height,Whole_weight,Shell_weight,i,m,f)

                                if output == 1:
                                    st.markdown(safe_html,unsafe_allow_html=True)
                                elif output == 2:
                                    st.markdown(warn_html,unsafe_allow_html=True)
                                elif output == 3:
                                    st.markdown(danger_html,unsafe_allow_html=True)
                with right_column:

                    clear = st.form_submit_button(label="Clear")


                    if clear:

                       st.empty()



    if __name__=='__main__':
            main()

elif choose == "Accurancy":
        #Add a file uploader to allow users to upload their project plan file
        def main():
            with st.container():
                st.subheader("Original Dataset")
                if st.button("TrainOrg"):
                    ee_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Mean Absolute Error :</h4>
                       </div>
                    """
                    aa_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Mean Squared Error :</h4>
                       </div>
                    """
                    bb_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Accuracy :</h4>
                       </div>
                    """
                    left_column,right_column = st.columns(2)
                    with left_column:


                        data=pd.read_csv('BeforeAccurancy.csv')


                        X = data.drop(['Age','Rings'], axis = 1)
                        y = data['Age']

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
                        X_train = X_train.values
                        X_test = X_test.values

                        y_train = y_train.values
                        y_test = y_test.values
                        classifiers =RandomForestClassifier()
                        classifiers.fit(X_train, y_train)
                        training_score = cross_val_score(classifiers, X_train, y_train, cv=5)

                        st.write(bb_html,unsafe_allow_html=True)
                        st.write(str(training_score.mean()))

                        y_pred = classifiers.predict(X_test)
                        mae=mean_absolute_error(y_test,y_pred);
                        st.write(ee_html,unsafe_allow_html=True)
                        st.write(str(mae))
                            #RMSE
                        rmse = math.sqrt(mean_squared_error(y_test,y_pred))
                        st.write(aa_html,unsafe_allow_html=True)
                        st.write(str(rmse))



                    with right_column:
                        image = Image.open('accuracy1.png')
                        st.image(image,width=500)
            st.write("----")
            with st.container():
                st.subheader("After Preprocessing")
                if st.button("Train"):
                    cc_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Mean Absolute Error :</h4>
                       </div>
                    """
                    dd_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Mean Squared Error :</h4>
                       </div>
                    """
                    ff_html="""  
                      <div style="background-color:#ccff99; padding:10px >
                       <h4 style="color:black ;text-align:center;"> Accuracy :</h4>
                       </div>
                    """
                    left_column,right_column = st.columns(2)

                    with left_column:


                        data=pd.read_csv('Accurancy.csv')


                        X = data.drop(['Age'], axis = 1)
                        y = data['Age']

                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
                        X_train = X_train.values
                        X_test = X_test.values

                        y_train = y_train.values
                        y_test = y_test.values
                        classifiers =RandomForestClassifier()
                        classifiers.fit(X_train, y_train)
                        training_score = cross_val_score(classifiers, X_train, y_train, cv=5)

                        st.write(ff_html,unsafe_allow_html=True)
                        st.write(str(training_score.mean()))

                        y_pred = classifiers.predict(X_test)
                        mae=mean_absolute_error(y_test,y_pred);
                        st.write(cc_html,unsafe_allow_html=True)
                        st.write(str(mae))
                        #RMSE
                        rmse = math.sqrt(mean_squared_error(y_test,y_pred))
                        st.write(dd_html,unsafe_allow_html=True)
                        st.write(str(rmse))
                        #Median Absolute error
                    with right_column:
                        image = Image.open('accuracy2.png')
                        st.image(image,width=500)




        if __name__=='__main__':
            main()
