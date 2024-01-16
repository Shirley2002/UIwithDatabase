import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
# import sklearn.tree as tree
# import sklearn.svm as svm
# from sklearn.model_selection import train_test_split
# from sklearn.feature_selection import RFE
# import sklearn.neighbors as knn
# from sklearn.linear_model import LogisticRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import *
# from sklearn.ensemble import StackingClassifier
# from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# from sklearn.feature_selection import SelectKBest, f_classif

# from sklearn.preprocessing import *
# from sklearn.pipeline import make_pipeline
# from sklearn.svm import SVR

import pickle as pickle



st.write(sklearn.__version__)


#####################

def set_state(stage):
    st.session_state.stage = stage

def handle_button():
    if st.session_state.stage == 0:
        FillData()
        Generate_Exercise_button()
    if st.session_state.stage == 1:
        Approve_Prescription_button()
    if st.session_state.stage == 2:
        Approved_Page()
    

    
if 'user_data' not in st.session_state:
    st.session_state.user_data = {} 

#UI

def predict_RF(HPT,DM,Smoking, BMI,Stress,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputRF= np.array([[HPT,DM,Smoking, BMI,Stress,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    #prediction = model_RF(inputRF)
    prediction = model_RF.predict(inputRF)

    return prediction

def predict_CR(HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputCR= np.array([[HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    #prediction = model_CR(inputRF)
    predictionCR = model_CR.predict(inputCR)

    return predictionCR


def predict_Weight(HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputWeight= np.array([[HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    #prediction = model_RF(inputRF)
    predictionWeight = model_Weight.predict(inputWeight)

    return predictionWeight

def FillData():
    #user_data = st.session_state.user_data
    st.session_state.user_data['HPT'] = st.radio(
    "Hypertension", ["Yes","No"],index=0, horizontal=True, key="1")
    #HPT =1 if user_data.get('HPT') == 'Yes' else 0

    st.session_state.user_data['Diabetes'] = st.radio(
    "Diabetic Mellitus", ["Yes","No"],index=0, horizontal=True, key="2")
    #DM =1 if user_data.get('Diabetes') == 'Yes' else 0

    st.session_state.user_data['Smoking'] = st.radio(
    "Smoking", ["Yes","No"],index=0, horizontal=True, key="3")
    #Smoking = 1 if user_data.get('Smoking') == 'Yes' else 0

    st.session_state.user_data['Stress'] = st.radio(
    "Stress", ["Yes","No"],index=0, horizontal=True, key="4")
    #Stress = 1 if user_data.get('Stress') =='Yes' else 0


    st.session_state.user_data['BMI'] = st.selectbox(
    "BMI",["Underweight", "Normal", "Overweight", "Obese"], key="5")
    #BMI = 0 if user_data.get('BMI')== 'Underweight' else 1 if user_data.get('BMI')=='Normal' else 2 if user_data.get('BMI')=="Overweight" else 3

    st.session_state.user_data['Hypertension'] = st.radio(
    "Hypertension", ["Yes","No"],index=0, horizontal=True, key="6")
    #Hypertension = 1 if user_data.get('Hypertension')=='Yes' else 0

    st.session_state.user_data['Ischemia'] = st.radio(
    "Ischemia", ["Yes","No"],index=0, horizontal=True, key="7")
    #Ischemia = 1 if user_data.get('Ischemia')=='Yes' else 0

    st.session_state.user_data['Dyslipidemia'] = st.radio(
    "Dyslipidemia", ["Yes","No"],index=0, horizontal=True, key="8")
    #Dyslipidemia = 1 if user_data.get('Dyslipidemia')=='Yes' else 0

    st.session_state.user_data['EF'] = st.number_input(
    "Ejection Fraction",0,1, key="9")
    #EjectionFraction= user_data.get('EF')

#st.button("Generate Exercise", on_click=set_state, args=[1])

    

def Generate_Exercise_button():
    if st.button("Generate Exercise"):
        user_data = st.session_state.user_data
        HPT =1 if user_data.get('HPT') == 'Yes' else 0
        DM =1 if user_data.get('Diabetes') == 'Yes' else 0
        Smoking = 1 if user_data.get('Smoking') == 'Yes' else 0
        Stress = 1 if user_data.get('Stress') =='Yes' else 0
        BMI = 0 if user_data.get('BMI')== 'Underweight' else 1 if user_data.get('BMI')=='Normal' else 2 if user_data.get('BMI')=="Overweight" else 3
        Hypertension = 1 if user_data.get('Hypertension')=='Yes' else 0
        Ischemia = 1 if user_data.get('Ischemia')=='Yes' else 0
        Dyslipidemia = 1 if user_data.get('Dyslipidemia')=='Yes' else 0
        EjectionFraction= user_data.get('EF')

    ##### Prediction ####
        RiskFactor= predict_RF(HPT,DM,Smoking,BMI,Stress,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
       
        
        if RiskFactor == 0:
            predict_Risk_Label = "low"
        elif RiskFactor == 1: 
            predict_Risk_Label = "moderate"
        else:
            predict_Risk_Label = "none"
        st.session_state.user_data['predict_risk_label'] = predict_Risk_Label

        CyclingResistance = predict_CR(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
        #Cycling_Resistance_Label = 1
        Cycling_Resistance_Label= 1 if CyclingResistance <=10 else 200
        # if CyclingResistance <=10:
        #     Cycling_Resistance_Label = 1
        # else:
        #     Cycling_Resistance_Label = 2
        st.session_state.user_data['Cycling_resistance_label']=Cycling_Resistance_Label


        Weight = predict_Weight(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
        #Weight_Label = 1 
        Weight_Label = 1 if Weight <= 4 else 23
        # if Weight <= 4:
        #     Weight_Label = 1
        # else:
        #     Weight_Label = 2
        st.session_state.user_data['Weight_label']= Weight_Label
        
        #predicted_risk_label = 'low'if RiskFactor ==0 else 'moderate'
        st.success("The risk factor is {}.".format(predict_Risk_Label))
        st.success("The suggested Cycling Resistance is {}.".format(Cycling_Resistance_Label))
        st.success("The suggested Weight is {}lb.".format(Weight_Label))

        
        st.button("Next Step", on_click=set_state, args=[1])    

def Approve_Prescription_button():
    user_data= st.session_state.user_data
    Prescription_text = "The risk factor is {}.  \n The suggested Cycling Resistance is {}.  \n The suggested weight is {}lb ".format(user_data.get('predict_risk_label'), user_data.get('Cycling_resistance_label'), user_data.get('Weight_label'))
    Final_Prescription= st.text_area("Generated prescription:", Prescription_text)

    st.session_state.user_data['Final_Prescription']= Final_Prescription

    st.button("Approve Prescription", on_click=set_state, args=[2])
    
def Approved_Page():
    user_data=st.session_state.user_data
    st.write("This is the final prescription:  \n  \n")
    st.success(user_data.get('Final_Prescription'))
   
#######################TEST####################
# def Generate_Exercise_button1():
#     HPT = 0
#     input_HPT= st.radio("Hypertension", ["Yes","No"],horizontal=True, key="1")
#     if input_HPT == "Yes":
#         HPT = 1
#     else:
#         HPT = 0
#     #st.session_state.user_data['Hypertension'] = HPT

#     DM = 0
#     input_DM= st.radio("Diabetic Mellitus",  ["Yes","No"],horizontal=True,key="2")
#     if input_DM == "Yes":
#         DM = 1
#     else:
#         DM = 0
#     st.session_state.user_data['DM'] == DM
#     Smoking = 0
#     input_Smoking= st.radio("Smoking", ["Yes","No"],horizontal=True, key="3")
#     if input_Smoking == "Yes":
#         Smoking = 1
#     else:
#         Smoking = 0

#     BMI = 0
#     input_BMI= st.selectbox("BMI",["Underweight", "Normal", "Overweight"], key="4")
#     if input_BMI == "Underweight":
#         BMI = 0
#     elif input_BMI == "Normal":
#         BMI = 1
#     else:
#         "Overweight"

#     Stress = 0    
#     input_Stress= st.radio("Stress", ["Yes","No"],horizontal=True, key="5")
#     if input_Stress == "Yes":
#         Stress = 1
#     else:
#         Stress = 0
           
#     Hypertension = 0
#     input_Hypertension= st.radio("Hypertension", ["Yes","No"],horizontal=True, key="6")
#     if input_Hypertension == "Yes":
#         Hypertension = 1
#     else:
#         Hypertension = 0

#     Ischemia = 0
#     input_Ischemia= st.radio("Ischemia", ["Yes","No"],horizontal=True, key="7")
#     if input_Ischemia == "Yes":
#         Ischemia = 1
#     else:
#         Ischemia = 0
    
#     Dyslipidemia = 0
#     input_Dyslipidemia= st.radio("Dyslipidemia",  ["Yes","No"],horizontal=True,key="8")
#     if Dyslipidemia == "Yes":
#         Dyslipidemia = 1
#     else:
#         Dyslipidemia = 0
    
#     input_EjectionFraction= st.number_input("Ejection Fraction",0,1, key="9")
#     EjectionFraction = input_EjectionFraction

     
#     if st.button("Generate Exercise1"): 
#          RiskFactor= predict_RF(HPT,DM,Smoking,BMI,Stress,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
#          if RiskFactor == 0:
#              predict_Risk_Label = "low"
#          elif RiskFactor == 1: 
#              predict_Risk_Label = "moderate"
#          else:
#              predict_Risk_Label = "high"
         
#          CyclingResistance = predict_CR(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
#          #Cycling_Resistance_Label = 1
#          if CyclingResistance <=10:
#              Cycling_Resistance_Label = 1
#          else:
#              Cycling_Resistance_Label = 2

#          Weight = predict_Weight(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
#          #Weight_Label = 1 
#          if Weight <= 4:
#              Weight_Label = 1
#          else:
#              Weight_Label = 2
#          #predicted_risk_label = 'low'if RiskFactor ==0 else 'moderate'
#          st.success("The risk factor is {}.".format(predict_Risk_Label))
#          st.success("The suggested Cycling Resistance is {}.".format(Cycling_Resistance_Label))
#          st.success("The suggested Weight is {}lb.".format(Weight_Label))
         
         #Prescription_text = "The risk factor is {}.  \n The suggested Cycling Resistance is {}.  \n".format(predict_Risk_Label, Cycling_Resistance_Label)

# def Edit_Prescription_button():
#         if st.button("Edit"):
#           user_edit = st.text_area("Edit the prescription:", Prescription_text)
#     #     if st.button("Submit"):

if __name__=='__main__':
    model_RF = pickle.load(open('stacked_model1.pkl','rb'))
    model_CR = pickle.load(open('stacked_model_CR.pkl', 'rb'))
    model_Weight = pickle.load(open('stacked_model_weight.pkl', 'rb'))
    st.header("Cardiac Rehabilitation Recommendation System")
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    handle_button()
    