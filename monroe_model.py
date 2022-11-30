
#pip install streamlit
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import streamlit as st
import os
from joblib import dump, load

#Load your trained models and csv file
model = load(str(str(os.getcwd())+'\DecisionTree_avg_prec_93.joblib'))
reg = load(str(str(os.getcwd())+'\Regression.joblib'))
df= pd.read_csv(str(str(os.getcwd())+'\Monroe_County_Single_Family_Residential__Building_'
                        'Assets_and_Energy_Consumption__2017-2019_prepped.csv'))

# Define your target values (y) and your features (df_un) often X
y='nyserda_energy_efficiency_program_participation_Yes';
df_dt=df.loc[:, df.columns != y]; # exclude the target value
df_dt= df_dt.loc[:, df_dt.columns != 'average_annual_total_energy_use_mmbtu']; #exclude the total energy column

# Create a dropdown box on your main paige
page = st.sidebar.selectbox("""
        Hello there! I’ll guide you!
         Please select model""",
         ["Main Page",
            "Classification",
            "Linear Regression",])

# Import an image on your main page
st.image('https://www.albireoenergy.com/wp-content/uploads/2017/02/NYSERDA.jpg')

# Main page
if page == "Main Page":
    st.subheader('About NYSERDA')
    st.write('Clean energy can power New York while protecting the environment. The New York State Energy Research and Development '
         'Authority, known as NYSERDA, promotes energy efficiency and the use of renewable energy sources. These efforts are key '
         'to developing a less polluting and more reliable and affordable energy system for all New Yorkers. Collectively, NYSERDA’s '
         'efforts aim to reduce greenhouse gas emissions, accelerate economic growth, and reduce customer energy bills.')
    st.subheader('About the Dataset')
    st.write('This aggregated and anonymized dataset of single-family residential building asset attributes and observed average '
         'annual energy consumption over the 2-year period from August 2017 through July 2019 is available for Monroe County.The '
         'dataset includes more than 55,000 properties from the study’s matched residential dataset that had sufficient data for '
         'calculation of average annual energy consumption ')

# Classificatie page
elif page == 'Classification':
    st.title('Can we predict if a househould is willing to contribute to NYSERDA?')
    st.write("Select the corresponding values on the left to do a prediction about the probability that the "
             "houshold will take part of the NSYERDA research.")
    year_built_range_var = st.sidebar.select_slider( ' year_built_range ' , ['<1945', '1945-1975', '1976-2000', '2000-2020'] )
    assessed_value_range_var = st.sidebar.select_slider( ' assessed_value_range ' , ['< 100','100 - 150',  '150 - 200', '200 - 300', '>300'] )
    number_of_stories_var = st.sidebar.number_input( ' number_of_stories ' , 1 , 3 )
    square_footage_range_var = st.sidebar.select_slider( ' square_footage_range ' , ['<= 1,500', '1500 - 2500', '>=2500'] )
    number_of_bedrooms_var = st.sidebar.select_slider( ' number_of_bedrooms ' , ['1 or 2', '3', '4 or more'] )
    total_number_of_bathrooms_var = st.sidebar.select_slider( ' total_number_of_bathrooms ' , ['1 or 1.5', '2 or 2.5', '3 or more'] )
    number_of_kitchens_var = st.sidebar.select_slider( ' number_of_kitchens ' , ['1 or less', '2 or more'] )
    number_of_fireplaces_var = st.sidebar.select_slider( ' number_of_fireplaces ' , ['0', '1 or more'] )
    ethnic_group_var = st.sidebar.select_slider( ' ethnic_group ' , ['other', 'Western European'] )
    number_of_occupants_var = st.sidebar.select_slider( ' number_of_occupants ' , ['Less than 3','3 or 4 occupants',  'More than 4'] )
    median_income_range_var = st.sidebar.select_slider( ' median_income_range ' , ['< 50', '50 - 100', '100 - 150', '> 150'] )
    average_annual_electric_use_mmbtu_var = st.sidebar.number_input( ' average_annual_electric_use_mmbtu ' , 0.0 , 594.99 )
    average_annual_gas_use_mmbtu_var = st.sidebar.number_input( ' average_annual_gas_use_mmbtu ' , 0.0 , 1287.75 )
    var = [year_built_range_var, assessed_value_range_var, number_of_stories_var,
           square_footage_range_var,  number_of_bedrooms_var,total_number_of_bathrooms_var,
           number_of_kitchens_var, number_of_fireplaces_var, ethnic_group_var, number_of_occupants_var,
           median_income_range_var, average_annual_electric_use_mmbtu_var,average_annual_gas_use_mmbtu_var]

    column= ['number_of_stories', 'average_annual_electric_use_mmbtu', 'average_annual_gas_use_mmbtu',
           'year_built_range', 'assessed_value_range',  'square_footage_range',
           'number_of_bedrooms', 'total_number_of_bathrooms', 'number_of_kitchens', 'number_of_fireplaces', 'ethnic_group',
           'number_of_occupants', 'median_income_range'];
    
    columnames_onehot = []
    with open("train_colnames.txt", "r") as f:
          for line in f:
                columnames_onehot.append(str(line.strip()))

    if st.sidebar.button('Predict'):
        dic={}
        for i in range(0,len(column)):
            dic[str(column[i])] = var[i]
        X_unseen = pd.DataFrame.from_dict([dic])
        X_unseen = pd.get_dummies(X_unseen).reindex(columns=columnames_onehot, fill_value=0)
        
        prediction=model.predict(np.array(X_unseen))[0]
        
        pred_prob = model.predict_proba(X_unseen)
        
        if prediction == 0:
            st.success(f"There is a chance of {pred_prob[0][0] * 100}% that the this household joins NSYERDA ")
        else:
            st.error(
                f"There is a chance of {pred_prob[0][1] * 100}% that this households does not join NSYERDA")

    st.sidebar.text("")

# Regression page
elif page == 'Linear Regression':
    st.title('Can we predict the total average energy used through a household based on the average annual electric usage?')
    average_annual_electric_use_mmbtu_var = st.number_input('Insert the average annual electric use in mmbtu ', 0.0, 594.99)

    if st.button('Predict'):
        prediction_lin= reg.predict(np.array(average_annual_electric_use_mmbtu_var).reshape(-1,1))
        st.write('The prediction of average total energy use in mmbtu is between the {} '.format(
            np.round(prediction_lin[0] - 26.76), 2), 'and the {}' .format(np.round(prediction_lin[0] + 26.76), 2))
