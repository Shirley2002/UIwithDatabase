import streamlit as st
import pandas as pd
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy import text
import pickle as pickle
from streamlit import connections

##########
# Initialize connection.
conn = st.connection('mysql', type='sql')



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

def predict_RF(HPT,DM,Smoking,BMI,Stress,Ischemia,Dyslipidemia,EjectionFraction, FH,EH,Diet,Alcohol,Balance,Walk,Gait):
    inputRF= np.array([[HPT,DM,Smoking, BMI,Stress, Ischemia,Dyslipidemia,EjectionFraction,FH,EH,Diet,Alcohol,Balance,Walk,Gait]])

    prediction = model_RF.predict(inputRF)

    return prediction

def predict_CR(Smoking,BMI,Ischemia,Dyslipidemia,EjectionFraction,Diet,Balance,Walk,Gait):
    inputCR= np.array([[Smoking,BMI,Ischemia,Dyslipidemia,EjectionFraction,Diet,Balance,Walk,Gait]])

    predictionCR = model_CR.predict(inputCR)

    return predictionCR


def predict_Weight(HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputWeight= np.array([[HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    predictionWeight = model_Weight.predict(inputWeight)

    return predictionWeight

def FillData():

    col1, col2 =st.columns(2)
    with col1:
        st.session_state.user_data['HPT'] = st.radio(
        "Hypertension", ["Yes","No"],index=0, horizontal=True, key="1")

        st.session_state.user_data['Diabetes'] = st.radio(
        "Diabetic Mellitus", ["Yes","No"],index=0, horizontal=True, key="2")

        st.session_state.user_data['Smoking'] = st.radio(
        "Smoking", ["Yes","No"],index=0, horizontal=True, key="3")

        st.session_state.user_data['Stress'] = st.radio(
        "Stress", ["Yes","No"],index=0, horizontal=True, key="4")

        st.session_state.user_data['BMI'] = st.selectbox(
        "BMI",["Underweight", "Normal", "Overweight", "Obese"], key="5")

        st.session_state.user_data['Hypertension'] = st.radio(
        "Posture Balance", ["Yes","No"],index=0, horizontal=True, key="6")

        st.session_state.user_data['Ischemia'] = st.radio(
        "Ischemia", ["Yes","No"],index=0, horizontal=True, key="7")

        st.session_state.user_data['Dyslipidemia'] = st.radio(
        "Dyslipidemia", ["Yes","No"],index=0, horizontal=True, key="8")

        st.session_state.user_data['EF'] = st.number_input(
        "Ejection Fraction",0,1, key="9")

    with col2:
        #FH,EH,Diet,Alcohol,Balance,Walk,Gait

        st.session_state.user_data['FH'] = st.radio(
        "Family History", ["Yes","No"],index=0, horizontal=True, key="10")

        st.session_state.user_data['EH'] = st.radio(
        "Exercise Habit", ["Yes","No"],index=0, horizontal=True, key="11")

        st.session_state.user_data['Diet'] = st.radio(
        "Undergoing Diet Control", ["Yes","No"],index=0, horizontal=True, key="12")

        st.session_state.user_data['Alcohol'] = st.radio(
        "Consumes Alcohol", ["Yes","No"],index=0, horizontal=True, key="13")
        
        st.session_state.user_data['Balance'] = st.radio(
        "Body Balancing", ["Yes","No"],index=0, horizontal=True, key="14")

        st.session_state.user_data['Walk'] = st.radio(
        "Able to walk", ["Yes","No"],index=0, horizontal=True, key="15")

        st.session_state.user_data['Gait'] = st.radio(
        "Gait", ["Normal","Abnormal"],index=0, horizontal=True, key="16")
    

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
        EjectionFraction= 0 if user_data.get('EF')>=0.5 else 1 if 0.41<user_data.get('EF')<0.49 else 2

         #FH,EH,Diet,Alcohol,Balance,Walk,Gait
        
        FH = 1 if user_data.get('FH')=='Yes' else 0
        EH = 1 if user_data.get('Exercise Habit')=='Yes' else 0
        Diet = 1 if user_data.get('Ischemia')=='Yes' else 0
        Alcohol = 1 if user_data.get('Alcohol')=='Yes' else 0
        Balance = 1 if user_data.get('Balance')=='Yes' else 0
        Walk = 1 if user_data.get('Walk')=='Yes' else 0
        Gait = 1 if user_data.get('Gait')=='Normal' else 0




    ##### Prediction ####
        RiskFactor= predict_RF(HPT,DM,Smoking,BMI,Stress,Ischemia,Dyslipidemia,EjectionFraction,FH,EH,Diet,Alcohol,Balance,Walk,Gait)
       
        
        if RiskFactor == 0:
            predict_Risk_Label = "low"
        elif RiskFactor == 1: 
            predict_Risk_Label = "moderate"
        else:
            predict_Risk_Label = "none"
        st.session_state.user_data['predict_risk_label'] = predict_Risk_Label

        CyclingResistance = predict_CR(Smoking,BMI,Ischemia,Dyslipidemia,EjectionFraction,Diet,Balance,Walk,Gait)
        Cycling_Resistance_Label= 1 if CyclingResistance<2 else 4 if CyclingResistance<3 else 3 if CyclingResistance<4 else 4 if CyclingResistance<5 else 6

        st.session_state.user_data['Cycling_resistance_label']=Cycling_Resistance_Label


        Weight = predict_Weight(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
        Weight_Label = 1.1 if Weight <= 6 else 2.2
        st.session_state.user_data['Weight_label']= Weight_Label
        

        st.success("The risk factor is {}.".format(predict_Risk_Label))
        st.success("The suggested Cycling Resistance is {}.".format(Cycling_Resistance_Label))
        st.success("The suggested Weight is {}lb.".format(Weight_Label))

        
        st.button("Next Step", on_click=set_state, args=[1])    

def Approve_Prescription_button():
    user_data= st.session_state.user_data
    Prescription_text = "The risk factor is {}.  \n The suggested Bike Resistance is {}.  \n The suggested weight is {}lb ".format(user_data.get('predict_risk_label'), user_data.get('Cycling_resistance_label'), user_data.get('Weight_label'))
    Final_Prescription= st.text_area("Generated prescription:", Prescription_text)
    PatientID= st.text_input("Patient ID:")
    Approveby= st.text_input("Approved by:")

    st.session_state.user_data['Final_Prescription']= Final_Prescription
    st.session_state.user_data['PatientID']= PatientID
    st.session_state.user_data['Approveby']= Approveby


    st.button("Approve Prescription", on_click=set_state, args=[2])
    
def Approved_Page():
    user_data=st.session_state.user_data
    st.write("This is the final prescription:  \n  \n")
    st.success(user_data.get('Final_Prescription'))


    with conn.session as session:
        # session.execute("INSERT INTO prescription (HPT, Diabetes) values ('{user_data.get('predict_risk_label')}', '{(user_data.get('Cycling_resistance_label'))};')'''))
        query = text("INSERT INTO prescription2 (PatientID, Prescription, ApprovedBy) VALUES (:PatientID, :FinalPrescription, :ApprovedBy);")
        session.execute(query, {"PatientID": user_data.get('PatientID'),"FinalPrescription": user_data.get('Final_Prescription'), "ApprovedBy": user_data.get('Approveby')})
        # session.execute("INSERT INTO prescription1 (PatientID) VALUES (:HPT);",{"HPT": user_data.get('predict_risk_label')})
        session.commit()



if __name__=='__main__':
    model_RF = pickle.load(open('model_RF.pkl','rb'))
    model_CR = pickle.load(open('model_BikeRes.pkl', 'rb'))
    model_Weight = pickle.load(open('stacked_model_weight.pkl', 'rb'))
    st.header("Cardiac Rehabilitation Recommendation System")
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    handle_button()
    