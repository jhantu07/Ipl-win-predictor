import pickle
import pandas as pd
import streamlit
import streamlit as stl

teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('pipe.pkl', 'rb'))
stl.title('IPL WIN PREDICTOR')
col1, col2 = stl.columns(2)
with col1:
    batting_team = stl.selectbox('select the batting team', teams)
with col2:
    bowling_team = stl.selectbox('select the bowling team', teams)
selected_city = stl.selectbox('select host city',sorted(cities))

target= streamlit.number_input('Target')

col3, col4, col5= stl.columns(3)

with col3:
    score = stl.number_input('Score')
with col4:
    overs = stl.number_input('overs completed')
with col5:
    wickets = stl.number_input('Wickets out')

if stl.button('Predict probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],
                             'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
    stl.table(input_df)
    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    stl.text(batting_team + "- " + str(round(win*100)) + "%")
    stl.text(bowling_team + "- " + str (round(loss*100)) + "%")
