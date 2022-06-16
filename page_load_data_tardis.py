# Package imports
import streamlit as st
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from datetime import datetime
from datetime import date
from dateutil.relativedelta import relativedelta

from tardis_dev import datasets, get_exchange_details
import logging
import asyncio
import nest_asyncio
#nest_asyncio.apply()
import csv
from tardis_client import TardisClient, Channel
import os
import itertools
import requests
import json
import urllib

os.chdir("/Users/manu/Quant Finance/OakTree & Lion")
api = "TD.0Hxv0-1z7UkLepBy.VyoKJypCdXnkRkk.w1vpWC0bFrlhZUI.Wkhd7l9rzLhCujN.fNAE0bw6m0Qc2Zc.0LgN"
#logging.basicConfig(level=logging.DEBUG)


def page_load_tardis():
    
    st.title('Get Data')
    
    exchanges = ['deribit', 'binance', 'binance-futures', 'coinbase', 'kraken', 'kraken-futures', 'ftx', 'bitmex']
    exchange_dropdown = st.selectbox('Pick your exchange', exchanges)
    
    def choose_exchange(exchanges): 
        for i in exchanges:
            if exchange_dropdown == i:
                exchange = i
                return exchange
            
    exchange = choose_exchange(exchanges)       
    
    exchange_checkbox = st.checkbox('Get exchange details')
    if exchange_checkbox:
        with st.spinner("Fetching data..."):  
            
#exchange = 'deribit'
 
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            exchange_details = get_exchange_details(exchange)
            
            #Gets CHANNEL
            channels = exchange_details['availableChannels']
            channels_dropdown = st.selectbox('Pick your channel', channels)
            
            def choose_channel(channels):
                for i in channels:
                    if channels_dropdown == i:
                        channel = i
                        return channel
                    
            channel = choose_channel(channels)
    
    #channel_checkbox = st.checkbox('Get channel details')
    #if channel_checkbox:
        #with st.spinner("Fetching data..."):

#channel = 'trades'
    
            #Turn to dataframe
            df_exchange_details = pd.DataFrame(exchange_details["datasets"]['symbols'])
            
            today = date.today()
            today = pd.Timestamp(today)
            six_months_ago = today.date() - pd.DateOffset(months = 6)
            df_exchange_details['availableSince'] = pd.to_datetime(df_exchange_details['availableSince'])
            df_exchange_details['availableTo'] = pd.to_datetime(df_exchange_details['availableTo'])
            df_exchange_details = df_exchange_details[(df_exchange_details["availableSince"] > six_months_ago.tz_localize('utc'))]
            
            
            #Gets CURRENCY
            df_exchange_details['currency'] = df_exchange_details['id'].str.split('_').str[0]
            df_exchange_details['currency'] = df_exchange_details['currency'].str.split('-').str[0]
            df_exchange_details = df_exchange_details[df_exchange_details['currency'].str.len() < 5]
            
            currencies = list(set(list(df_exchange_details['currency'])))
            currency_dropdown = st.selectbox('Pick your currency', currencies)

            def choose_currency(currencies):
                for i in currencies:
                    if currency_dropdown == i:
                        currency = i
                        return currency
                    
            currency = choose_currency(currencies)
    
    #currency_checkbox = st.checkbox('Get currency details')
    #if currency_checkbox:
        #with st.spinner("Fetching data..."):

#currency = 'ETH'  
  
            df_exchange_details = df_exchange_details[df_exchange_details['currency'] == currency]

            #Gets TYPE
            types = list(set(list(df_exchange_details['type'])))
            types_dropdown = st.selectbox('Pick your instrument type', types)
            
            def choose_type(types):
                for i in types:
                    if types_dropdown == i:
                        single_type = i
                        return single_type
                    
            single_type = choose_type(types)
    
    #type_checkbox = st.checkbox('Get instrument type details')
    #if type_checkbox:
        #with st.spinner("Fetching data..."):
                                
#single_type = 'future'

            #Gets ids available based on type
            ids = list(set(list(df_exchange_details['id'][df_exchange_details['type'] == single_type])))
            id_options = ids
            id_options.append('All')
            id_options.sort()
            id_options = st.multiselect('Pick your instruments', id_options)
            if id_options[0] == 'All':
                id_options = ids[1:]

#id_options = list(['ETH-10APR21-2220-C', 'ETH-10FEB22-3800-C'])
#symbol = 'ETH-30DEC22'       
           
            since = []
            to = []
            for i in id_options:
                since.append(list(df_exchange_details['availableSince'][df_exchange_details['id'] == i]))
                to.append(list(df_exchange_details['availableTo'][df_exchange_details['id'] == i]))
            
            new_since = []
            new_to = []
            
            slider_keys = list(range(0, len(id_options)))

            for symbol_id, start, end, keys in zip(id_options, since, to, slider_keys):
                st.write(symbol_id + ' ' + channel + ' data')
                
                format = 'YY/MM/DD - hh:mm:ss'  
                date_start = datetime.timestamp(pd.to_datetime(start[0]))
                date_start = datetime.fromtimestamp(date_start)
                date_end = datetime.timestamp(pd.to_datetime(end[0]))
                date_end = datetime.fromtimestamp(date_end)

                slider = st.slider('Select date', min_value=date_start, value=(date_start, date_end), max_value=date_end , format=format, key=(keys))
                st.write("Start time:", slider[0])
                st.write("End time:", slider[1])
                new_since.append(slider[0])
                new_to.append(slider[1])
     
                
            total_time = []
            for start, end in zip(new_since, new_to):
                total_time.append((end-start).days * 6)
            
            st.write("This data should take about: " + str(sum(total_time)) + " seconds to complete.")        
            st.write("That is: " + str(sum(total_time)/60) + " minutes.")
            st.write("Or: " + str(sum(total_time)/60/60) + " hours.")
     
                

#data_options = ['timestamp', 'instrument_name', 'asks', 'bids']
    if st.button('Get Data'):
        with st.spinner("Fetching data..."):
            
            path = exchange + '/' + currency + '/' + single_type + '/' + channel
            
            st.write("Starting download now...")

            for symbol, start, end in zip(id_options, new_since, new_to):

                st.write("This single file should take about: " + str((end-start).days * 6) + " seconds.")        
                st.write('Downloading: ' + symbol + ' ' + channel + ' data. From ' + str(start)[:10] + ' until ' + str(end)[:10] + '.')

                file = symbol + '.csv'

                async def get_data():
                                    
                    tardis_client = TardisClient(api_key=api)
                        
                    messages = tardis_client.replay(
                        exchange=exchange,
                        from_date=str(start)[:10],
                        to_date=str(end)[:10],
                        filters=[Channel(name=channel, symbols=[symbol])],
                    )
                    
                    if(os.path.exists(path)):
                       st.write('Path exists. Finding file now...')
                       if(os.path.exists(path + '/' + file)):
                           st.write('File found too. Removing it and re-writing...')
                           os.remove(path + '/' + file)
                       else:
                           st.write('File not found. Creating it now...')
                    else:
                       st.write('Path and file do not exists. Creating them now...')
                       os.makedirs(path)   
                    
                    
                    rows = []
                    async for local_timestamp, message in messages:
                        data = message["params"]["data"]
                        if type(data) == list:
                            data = data[0]
                        else:
                            pass
                        dicts = {}
                        for i in data:
                            dicts.update({i: data[f'{i}'.format(i)]})
                        rows.append(dicts)
                        
                    df = pd.DataFrame.from_dict(rows) 
                    df.to_csv(path + '/' + file, index = False, header=False)
                    
                asyncio.run(get_data())
                
                st.write("finished") 
                    



