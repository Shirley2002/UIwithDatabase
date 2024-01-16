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


# # Perform query.
# df = conn.query('SELECT * from prescription1;', ttl=600)

# # Print results.
# for row in df.itertuples():
#     st.write(f"{row.PrescriptionID} has a :{row.PatientID}:")

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

    prediction = model_RF.predict(inputRF)

    return prediction

def predict_CR(HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputCR= np.array([[HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    predictionCR = model_CR.predict(inputCR)

    return predictionCR


def predict_Weight(HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction):
    inputWeight= np.array([[HPT,DM,Smoking, BMI,Hypertension, Ischemia,Dyslipidemia,EjectionFraction]])

    predictionWeight = model_Weight.predict(inputWeight)

    return predictionWeight

def FillData():
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
    "Hypertension", ["Yes","No"],index=0, horizontal=True, key="6")

    st.session_state.user_data['Ischemia'] = st.radio(
    "Ischemia", ["Yes","No"],index=0, horizontal=True, key="7")

    st.session_state.user_data['Dyslipidemia'] = st.radio(
    "Dyslipidemia", ["Yes","No"],index=0, horizontal=True, key="8")

    st.session_state.user_data['EF'] = st.number_input(
    "Ejection Fraction",0,1, key="9")

    

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
        Cycling_Resistance_Label= 1 if CyclingResistance <=10 else 200

        st.session_state.user_data['Cycling_resistance_label']=Cycling_Resistance_Label


        Weight = predict_Weight(HPT,DM,Smoking,BMI,Hypertension,Ischemia,Dyslipidemia,EjectionFraction)
        Weight_Label = 1 if Weight <= 4 else 23
        st.session_state.user_data['Weight_label']= Weight_Label
        

        st.success("The risk factor is {}.".format(predict_Risk_Label))
        st.success("The suggested Cycling Resistance is {}.".format(Cycling_Resistance_Label))
        st.success("The suggested Weight is {}lb.".format(Weight_Label))

        
        st.button("Next Step", on_click=set_state, args=[1])    

def Approve_Prescription_button():
    user_data= st.session_state.user_data
    Prescription_text = "The risk factor is {}.  \n The suggested Cycling Resistance is {}.  \n The suggested weight is {}lb ".format(user_data.get('predict_risk_label'), user_data.get('Cycling_resistance_label'), user_data.get('Weight_label'))
    Final_Prescription= st.text_area("Generated prescription:", Prescription_text)
    PatientID= st.text_input("Patient ID:")

    st.session_state.user_data['Final_Prescription']= Final_Prescription
    st.session_state.user_data['PatientID']= PatientID

    st.button("Approve Prescription", on_click=set_state, args=[2])
    
def Approved_Page():
    user_data=st.session_state.user_data
    st.write("This is the final prescription:  \n  \n")
    st.success(user_data.get('Final_Prescription'))


    with conn.session as session:
        # session.execute("INSERT INTO prescription (HPT, Diabetes) values ('{user_data.get('predict_risk_label')}', '{(user_data.get('Cycling_resistance_label'))};')'''))
        query = text("INSERT INTO prescription1 (PatientID, Prescription) VALUES (:PatientID, :FinalPrescription);")
        session.execute(query, {"PatientID": user_data.get('PatientID'),"FinalPrescription": user_data.get('Final_Prescription')})
        # session.execute("INSERT INTO prescription1 (PatientID) VALUES (:HPT);",{"HPT": user_data.get('predict_risk_label')})
        session.commit()



if __name__=='__main__':
    model_RF = pickle.load(open('stacked_model1.pkl','rb'))
    model_CR = pickle.load(open('stacked_model_CR.pkl', 'rb'))
    model_Weight = pickle.load(open('stacked_model_weight.pkl', 'rb'))
    st.header("Cardiac Rehabilitation Recommendation System")
    if 'stage' not in st.session_state:
        st.session_state.stage = 0
    handle_button()
    