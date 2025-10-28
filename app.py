import streamlit as st
import pandas as pd
from apputil import GroupEstimate

st.write("""
# Week 9: Group Estimate
This app demonstrates a simple use of the `GroupEstimate` class.
Enter an integer below, and the model will "predict" based on that value.
""")

# Input section
amount = st.number_input("Exercise Input:", step=1, format="%d", value=0)
estimate_type = st.selectbox("Select estimate type:", ['mean', 'median'])

if st.button("Run GroupEstimate"):
    # Create a simple dummy dataset for demo
    df = pd.DataFrame({
        'group': ['A', 'A', 'B', 'B', 'C', 'C'],
        'value': [10, 12, 20, 22, 30, 28]
    })

    X = df[['group']]
    y = df['value']

    # Initialize your model
    ge = GroupEstimate(estimate=estimate_type)
    ge.fit(X, y, default_category='group')

    # Simulate predicting using the user input
    # We'll just pretend the user input belongs to one of the groups
    X_test = pd.DataFrame({'group': ['A']})  # You can change 'A' dynamically later
    prediction = ge.predict(X_test)

    st.success(f"The prediction for group 'A' using {estimate_type} is: {prediction[0]}")
