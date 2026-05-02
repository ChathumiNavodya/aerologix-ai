from turtle import st
from modules.api import get_live_flights

if st.button("Load Live Flights"):
    data = get_live_flights()
    st.json(data)