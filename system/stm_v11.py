"""
Created on Tue Jan 23 10:25:46 2024

@author: lhy
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import engines.analysis as ana
import engines.arima as ari
import engines.xgboost as xgb
import engines.Lstm as lstm
import engines.GRU as gru
from engines.GRU import RNNModel
import engines.Randomforest as rf
import engines.LightGBM as lg
import os
import pickle
# 定义页面标识
page = st.sidebar.selectbox('Choose your page', ['Main Page', 'Visualization','Basic Analysis', 'Prediction',"Risk Analysis"])

current_dir = os.path.dirname(__file__)
relative_path = os.path.join(current_dir, 'dataset/data.csv')
data = pd.read_csv(relative_path)

# A sample feature mapping

feature_map_total = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col not in ['country', 'year',"id"]]}
feature_revise_map_total = {v:k for k,v in feature_map_total.items()}
feature_map_oil = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['oil_product',
               'oil_price','oil_value','oil_exports','oil_pro_person','oil_val_person']]}
feature_revise_map_oil = {v:k for k,v in feature_map_oil.items()}
feature_map_gas = {v:"".join([f"{i.capitalize()} " for i in v.split("_")])[0:-1]
               for v in [col for col in data.columns if col in ['gas_product',
               'gas_price','gas_value','gas_exports','gas_pro_person','gas_val_person']]}
feature_revise_map_gas = {v:k for k,v in feature_map_gas.items()}

oil_features = ['oil_product','oil_price','oil_value','oil_exports',
                'oil_pro_person','oil_val_person']
gas_features = ['gas_product','gas_price','gas_value','gas_exports',
                'gas_pro_person','gas_val_person']

if page == 'Main Page':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# ENERGY PLATFORM")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Main Page')
    st.write('Content for Main Page')
    image_path = 'system/pictures/mainpage.jpeg'
    st.image(image_path, caption='Oil & Gas', use_column_width=True)

elif page == 'Visualization':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# Visualization")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Visualization')
    energy_option = st.sidebar.radio('1.Energy Options', ['Oil', 'Gas'])
    
    if energy_option == 'Oil':
        feature_map = feature_map_oil
        features = oil_features
        feature_revise_map = feature_revise_map_oil
    elif energy_option == 'Gas':
        feature_map = feature_map_gas
        features = gas_features
        feature_revise_map = feature_revise_map_gas
        
    cities = data["country"].unique().tolist()
    
    min_year, max_year = int(data['year'].min()), int(data['year'].max())
    start_year = st.slider("Choose start year", min_year, max_year, min_year)
    end_year = st.slider("Choose end year", min_year, max_year, max_year)
    if end_year < start_year:
            st.error("The end year must later than the start year!")
    else:
        patterns = ['Mutiple countries with one feature','Mutiple features in one country']
        pattern_option = st.selectbox('Please select a pattern',patterns)
        
        if pattern_option == 'Mutiple countries with one feature':
            cities_option = st.multiselect('Please select one or more countries',cities)
            feature_option = st.selectbox('Please select one feature',[feature_map[col] for col in data.columns if col in features])
            feature_option = feature_revise_map[feature_option]
            lines, = ana.trend(data,cities_option,feature_option,int(start_year),int(end_year))
            if cities_option:
                df = pd.DataFrame(
                        np.array([line['y'] for line in lines]).T,
                        columns = cities_option,
                        index = lines[0]['x'] )
                st.title(feature_map[feature_option]+ ' of Different Countries')   
                st.line_chart(df)
                
        elif pattern_option == 'Mutiple features in one country':
            city_option = st.selectbox('Please select one country',cities)
            features_option = st.multiselect('Please select one or more features',[feature_map[col] for col in data.columns if col in features])
            feature_option = [feature_revise_map[feature] for feature in features_option]
            lines, = ana.corr_features_cities(data,city_option,feature_option,int(start_year),int(end_year))
            if features_option:
                df = pd.DataFrame(
                        np.array([line['y'] for line in lines]).T,
                        columns = features_option,
                        index = lines[0]['x'] )
                st.title('Different Features of '+ city_option)   
                st.line_chart(df)

elif page == 'Basic Analysis':
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# DATA ANLYSIS")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    st.sidebar.write('Sidebar for Analysis')
    
    # 添加用于选择是否检测和剔除outliers的选项
    def outlier_detect(data,city,feature):
        filtered_data = data[(data['country'] == city)]
        fig,ax = plt.subplots(figsize = (10,6))
        # 绘制箱线图
        ax.boxplot(filtered_data[feature])
        ax.set_xlabel("feature")
        ax.set_ylabel("value")
        ax.set_title("Box Plot")
        return fig

    outlier_option = st.sidebar.radio('Outlier Options', ['Detect', 'Drop'])
    if outlier_option == 'Detect':
        st.title("Detecting outliers...")
        cities_option = st.selectbox("Please select one or more countries", data['country'].unique())
        features_option = st.selectbox("Choose one feature", [feature_map_total[col] for col in data.columns if col not in ['country', 'year',"id"]])
        fig = outlier_detect(data,cities_option,feature_revise_map_total[features_option])
        if features_option:
            st.pyplot(fig)

    # 这里添加检测outliers的代码
    elif outlier_option == 'Drop':
        data = data.bfill()
        st.write('Outliers have been dropped...')
        # 这里添加剔除outliers的代码

        #patterns = ['correlation with cities','correlation with features',"seasonal trend decomposition"]
        patterns = ['Correlation with countries','Correlation with features']
        pattern_option = st.selectbox('Please select a pattern',patterns)
        
        if pattern_option == 'Correlation with countries':
            st.title("Correlation with countries")
            # 用户输入
            cities_option = st.multiselect("Please select one or more countries", data['country'].unique(), key="cities")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("Choose start year", min_year, max_year, min_year)
            end_year = st.slider("Choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                feature_option = st.multiselect("Choose one or more features", [feature_map_total[col] for col in data.columns if col not in ['country', 'year','id','population']],key="feature_names")

            # 如果用户已经做出选择，则显示图表
            if cities_option and feature_option:
                bars,graph_params = ana.corr_cities(data, cities_option, [feature_revise_map_total[feature] for feature in feature_option], start_year, end_year)
                df = pd.DataFrame(
                           np.array([bar['y'] for bar in bars]).T,
                           columns = [bar['label'] for bar in bars],
                           index = graph_params["set_xticks"][1])
                st.title('correlation with cities')   
                st.bar_chart(df)
                
        elif pattern_option == 'Correlation with features':
            st.title("Correlation with features")

            # 确保年份选择逻辑正确
            min_year, max_year = int(data['year'].min()), int(data['year'].max())
            start_year = st.slider("Choose start year", min_year, max_year, min_year)
            end_year = st.slider("Choose end year", min_year, max_year, max_year)

            # 确保用户不能选择结束年份小于开始年份
            if end_year < start_year:
                st.error("The end year must later than the start year!")
            else:
                features_option = st.multiselect("Choose one or more features", [col for col in data.columns if col not in ['country', 'year','id']])


            fig = ana.corr_features(data,features_option,start_year,end_year)
            if features_option:
                st.pyplot(fig)

elif page == 'Prediction':
    # Logo and Navigation
    col1, col2, col3 = st.columns((1, 4, 1))
    with col2:
        st.markdown("# Prediction")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    with open('./system/engines/tuple_dict.pkl', 'rb') as f:
        country_dict = pickle.load(f)
    countries = list(country_dict.keys())
    model_option = st.sidebar.radio('Model Options', ['Linear','Decision Tree','Random Forest','LightGBM','RNN','GRU','LSTM','XGBoost','Arima'])
    default_countries_index = countries.index('United States')
    country_option = st.selectbox('Please select one country',countries,index = default_countries_index)
    features_trained = country_dict[country_option]
    feature_option = st.selectbox('Please select one feature',[feature_map_total[col] for col in features_trained])
    feature_option = feature_revise_map_total[feature_option]
    if model_option == 'Linear':
        st.write("Linear Model Result:")
    elif model_option == 'Decision Tree':
        st.write("Decision Tree Model Result:")
    elif model_option == 'Random Forest':
        st.write("Random Forest Model Result:")
        try:
            plt,pred,years = rf.randomforest_func(data,country_option,feature_option,0)
            data = {}
            df = pd.DataFrame(data)
            for i in range(len(years) - 5,len(years)):
                y = years[i].strftime("%Y-%m-%d")
                y = y[:4]
                df[y] = pred[i]
            st.write(df)
            st.pyplot(plt)
        except:
            st.write("No Such Model!")
    elif model_option == 'LightGBM':
        st.write('LightGBM Model Result')
        plt,pred,years = lg.lightgbm_func(data,country_option,feature_option,0)
        data = {}
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(plt)
    elif model_option == 'RNN':
        st.write("RNN Model Result:")
        fig = gru.RNN_model(country_option,feature_option)
        st.pyplot(fig)
    elif model_option == 'GRU':
        st.write("GRU Model Result:")
        fig = gru.GRU_model(country_option,feature_option)
        st.pyplot(fig)
    elif model_option == 'LSTM':
        st.write("LSTM Model Result:")
        fig,pred,years = lstm.predict(data,country_option,feature_option,0)
        data = {}               
        df = pd.DataFrame(data)
        for i in range(len(years) - 5,len(years)):
            y = years[i].strftime("%Y-%m-%d")
            y = y[:4]
            df[y] = pred[i]
        st.write(df)
        st.pyplot(fig)
    elif model_option == 'XGBoost':
        st.write("XGBoost Model Result:")
        try:
            plt,pred,years = xgb.xgboost_func(data,country_option,feature_option,0)
            data = {}
            df = pd.DataFrame(data)
            for i in range(len(years) - 5,len(years)):
                y = years[i].strftime("%Y-%m-%d")
                y = y[:4]
                df[y] = pred[i]
            st.write(df)
            st.pyplot(plt)
        except:
            st.write("No Such Model!")
    elif model_option == 'Arima':
        st.title("ARIMA")
        # 用户输入
        cities_option = st.selectbox("Please select one or more countries", data['country'].unique(), key="cities")

        # 确保年份选择逻辑正确
        min_year, max_year = int(data['year'].min()), int(data['year'].max())
        start_year = st.slider("Choose start year", min_year, max_year, min_year)
        end_year = st.slider("Choose end year", min_year, max_year, max_year)

        # 确保用户不能选择结束年份小于开始年份
        if end_year < start_year:
            st.error("The end year must later than the start year!")
        else:
            feature_option = st.selectbox("Choose one or more features", [col for col in data.columns if col not in ['country', 'year',"id"]],key="feature_names")

        # 如果用户已经做出选择，则显示图表
        if cities_option and feature_option:
            figs,result =ari.pred_arima(data, cities_option, feature_option, start_year, end_year)
            st.pyplot(figs[0])
            st.pyplot(figs[1])
            st.pyplot(figs[2])
            st.pyplot(figs[3])
            st.write(result)




elif page == 'Risk Analysis':
    # Logo and Navigation
    col1,col2, col3 = st.columns((0.1,4,1))
    with col2:
        st.markdown("# Risk Transmission Analysis")
    with col3:
        st.markdown("[![GitHub](https://img.shields.io/badge/-GitHub-black?logo=github&style=flat-square)](https://github.com/msdm-ust/energyintel_data_platform)", unsafe_allow_html=True)
    analysis_option = st.sidebar.radio('Analytical Perspectives', ['Global','National'])

    if analysis_option == 'Global':
        energies = ['Oil',"Gas"]
        energy_option = st.selectbox('Energy Options',energies)
        
        if energy_option == 'Oil':
            figures = ['Total Volatility Spillovers',"Directional Volatility Spillovers(To all others)",
                         "Directional Volatility Spillovers(From all others)","Net Volatility Spillovers",
                         "Net Pairwise Volatility Spillovers"]
            figure_option = st.selectbox('Please select a pattern',figures)
            if figure_option=="Total Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/oil_1.png'
                st.image(image_path, caption='Total Volatility Spillovers', use_column_width=True)
                text = """
This chart shows the total volatility spillover from the stock markets of the six countries in the oil risk system plus the WTI crude oil futures price. Aggregate volatility spillover is how volatility in one market affects volatility in other markets. Here's how to analyze this data set:
1. **large increase at the beginning of 2020**.
   - At the beginning of 2020, the graph shows a significant peak in the aggregate volatility spillover. This could be related to the sharp decline in global oil demand, especially due to the global embargo and travel restrictions triggered by the COVID-19 pandemic.
During the same period, the failure of OPEC+ to agree on production cuts triggered a short-lived price war, further increasing market volatility.
2. **Volatility beyond 2020**.
   - After the peak of the COVID-19 pandemic, the Total Volatility Spillover Index showed an upward trend. This may reflect market uncertainty about the future recovery of oil demand, as well as ongoing concerns about the path of global economic recovery.
3. **Continuing upward trend**.
   - The overall upward trend in the Volatility Spillover Index from 2020 through 2024 may be related to several factors, including, but not limited to, the uncertain recovery of the global economy, the long-term impact of the energy transition on oil demand, and the impact of geopolitical tensions on the oil market.
4. **Impact of global factors**.
   - The chart may also reflect the impact of other global factors, such as changes in monetary policy, adjustments in trade policy, and uncertainty about major political events.
Market interactions.

The Total Volatility Spillover Index also shows the complexity of the interactions between the oil market and the stock markets of the six countries. Instability in one country's market can be quickly transmitted throughout the system, affecting oil futures prices.

"""
                st.write(text) 


            elif figure_option == "Directional Volatility Spillovers(To all others)":
                image_path = 'system/pictures/risk_analysis/oil_2.png'
                st.image(image_path, caption='Directional Volatility Spillovers(To all others)', use_column_width=True)
                text = """
In this chart, we observe the directional volatility spillovers from several countries (the United States, Britain, China, France, Italy, and Germany) and WTI crude oil futures prices to other markets. Directional volatility spillovers refer to how volatility in one market affects volatility in other markets. Below is an analysis of each market's situation:

1. **United States (America TO all others)**:
   - Volatility peaked in early 2020, likely due to the extreme uncertainty and turmoil in global financial markets caused by the COVID-19 pandemic.
   - Subsequently, volatility decreased, possibly reflecting the market gradually adapting to the pandemic's impact and the implementation of economic stimulus measures.

2. **Britain (Britain TO all others)**:
   - In 2020, volatility also increased, which may reflect the dual impact of Brexit uncertainty and the COVID-19 pandemic.
   - The determination of the subsequent Brexit agreement may have helped reduce volatility.

3. **China (China TO all others)**:
   - China's market volatility rose in 2020, possibly related to the initial uncertainty of the pandemic and disruptions to the global supply chain.
   - China's earlier success in controlling the pandemic may have led to a relatively stable influence of its market on others afterward.

4. **France (France TO all others)**:
   - Volatility in France continued to rise after 2020, likely reflecting the country's response to the COVID-19 pandemic and the impact of European economic recovery plans.

5. **Italy (Italy TO all others)**:
   - Similar to France, the rise in volatility could be related to the economic damage Italy suffered during the pandemic and policy expectations related to the European Recovery Fund.

6. **Germany (Germany TO all others)**:
   - Germany's market volatility was lower in 2020 but rose after 2022, possibly reflecting the country's stability and its influence in the global financial markets.

7. **WTI Crude Oil Futures (WTI TO all others)**:
   - At the beginning of 2020, the volatility of WTI rose sharply, obviously related to the turmoil in the global oil market, especially the decline in demand caused by the COVID-19 pandemic.
   - Subsequent volatility may be related to the gradual opening of the global economy and changes in the supply and demand dynamics of the oil market.

These volatility indicators provide a perspective on the interplay between different markets, showing the interconnectedness and fragility of global financial markets at different points in time. Each country's indicator is a measure of the impact of that country's market volatility on other global markets.
"""
                st.write(text)                

                

            elif figure_option == "Directional Volatility Spillovers(From all others)":
                image_path = 'system/pictures/risk_analysis/oil_3.png'
                st.image(image_path, caption='Directional Volatility Spillovers(From all others)', use_column_width=True)
                text = """
In this set of charts, we see directional volatility spillovers from other markets to each respective country's market (FROM all others). This is a measure of how volatility from other country markets affects the volatility of a specific country's market. Below is an analysis for each country and WTI crude oil futures:

1. **United States (America FROM all others)**:
   - Volatility rose in 2020, which could be due to the instability of other markets during the global financial crisis impacting the U.S. market.
   - Subsequent large fluctuations may reflect the sensitivity of the U.S. to global events and other market changes.

2. **Britain (Britain FROM all others)**:
   - The significant rise in 2020 may be related to the global uncertainty associated with the pandemic and increased uncertainties in the Brexit process.
   - The growing trend in volatility afterward could reflect the UK market's increasing sensitivity to volatility from other markets.

3. **China (China FROM all others)**:
   - The rise in volatility in 2020 could be related to the global market's reaction to the initial outbreak in China and the potential global economic impact.
   - Changes in volatility afterward might reflect China's recovery in the global economy and fluctuations.

4. **France (France FROM all others)**:
   - The increase in volatility in 2020 could be related to the impact of the pandemic and the uncertainties in European economic policies.
   - The continued upward trend in volatility may be due to the French market's high sensitivity to the dynamics of other markets.

5. **Italy (Italy FROM all others)**:
   - The sharp rise in volatility in 2020 reflects the profound impact of the pandemic in Italy, one of the early epicenters in Europe.
   - The subsequent reduction in volatility could be related to Italy's management of the pandemic and expectations around the European Recovery Fund.

6. **Germany (Germany FROM all others)**:
   - The changes in volatility might be associated with Germany's central role in the global and particularly European economy.
   - Fluctuations in volatility in 2020 might reflect the global uncertainties impacting the German market.

7. **WTI Crude Oil Futures (WTI FROM all others)**:
   - The peaks in volatility could be related to the extreme volatility in the global oil market in 2020.
   - The pandemic and other global economic factors could have led to WTI's market response to volatility in other markets.

Overall, these charts reflect how the pandemic and subsequent global market volatility have impacted each country's stock market and the WTI crude oil market. The response of each country's market depends not only on domestic factors but also on the sentiment of other country markets and the global economic environment.
"""
                st.write(text)


            elif figure_option == "Net Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/oil_4.png'
                st.image(image_path, caption='Net Volatility Spillovers', use_column_width=True)
                text = """
This set of charts represents the net volatility spillover, which is the spillover of volatility from one country's market to other markets minus the spillover of volatility from other markets to that country's market. If a country's net volatility spillover is positive, this means that it transmits more volatility to the outside world than it receives from the outside world; if it is negative, it receives more volatility than it transmits. Each chart is analyzed separately below:
1. **United States (NET America)**:
   - Net volatility spillovers are more balanced overall, showing roughly equal transmission and reception of volatility in the U.S. market.
   - The small ups and downs in 2020 may reflect a short-lived imbalance between the impact of global markets on the US in the early stages of the epidemic and the impact of US markets on the world.
2. **the United Kingdom (NET Britain)**:
   - The UK market shows a significant negative net volatility spillover in 2020, suggesting that the UK market was more affected by external markets during this period.
   - This could be linked to uncertainty over the Brexit process and the COVID-19 outbreak.
3. **China (net china)**:
   - Less volatility in 2020 suggests that the Chinese market remains relatively balanced between volatility transmission and reception in the global market.
   - Subsequent small increases may indicate that the Chinese market is beginning to transmit more of the external volatility.
4. **France (NET France)**:
   - An upward trend in the Volatility Spillover Index after 2020 could indicate that the French market is passing on more volatility to external markets.
5. **Italy (NET Italy)**:
   - Significantly negative values in 2020 indicate that Italy was more affected by external markets in the early stages of the epidemic.
   - The subsequent upward trend may reflect a gradual increase in the Italian market's exposure to external volatility.
6. **NET Germany**:
   - The more significant negative net volatility spillover in 2020 may be related to Germany's high sensitivity to global economic fluctuations as an export-oriented economy.
   - The rise in net volatility spillover thereafter suggests an increase in the German market's external exposure.
7. **WTI crude oil (net WTI)**:
   - In 2020, the net volatility spillover is positive, implying that WTI price volatility affects other markets more than other markets affect WTI.
   - This could be related to supply and demand dynamics and price wars in the global oil market.

These indices reflect the interactions between different markets. A positive value may indicate that the market has a stronger presence in the global financial system, while a negative value may indicate that the market is more vulnerable to external shocks.

"""
                st.write(text)


            elif figure_option == "Net Pairwise Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/oil_5.png'
                st.image(image_path, caption='Net Pairwise Volatility Spillovers', use_column_width=True)
                text = """
let's proceed with a qualitative interpretation based on the general trends observed in each pairwise comparison of the net volatility spillovers:

1.	**America and France**: 
   - The net spillover between America and France seems to fluctuate around a baseline without extreme peaks, suggesting a balanced exchange of volatility between the two markets over time.

2.	**America and Britain**: 
   - The interaction appears relatively stable with occasional fluctuations, indicating periods where one market may have influenced the other more, but without a clear long-term dominant direction.

3.	**America and Italy**: 
   - This pair shows some periods of heightened spillover from one market to the other, potentially reflecting specific economic or political events affecting the transatlantic dynamic.

4.	**America and China**: 
   - Fluctuations are evident, possibly reflecting changing economic relations, trade tensions, or policy announcements impacting the markets' interaction.

5.	**America and Germany**: 
   - The trend is relatively stable with minor peaks. Since both are major economies, their financial markets likely have a balanced influence on each other.

6.	**America and WTI**: 
   - Considering America's significant role in global oil markets, the interaction with WTI could reflect changes in energy policies, oil supply dynamics, or shifts in global demand.

7.	**France and Britain, France and Italy, France and China, France and Germany**: 
   - These plots show the interactions of France with other European countries and China. Variations might indicate regional economic developments, European Union policy decisions, or shifts in the global economic outlook.

8.	**France and WTI**: 
   - The relationship between French market volatility and WTI could be indicative of how global oil price changes impact the French economy, which may be sensitive to energy market shifts.

9.	**Britain and Italy, Britain and China, Britain and Germany**: 
   - The UK's interactions with these markets may have been influenced by Brexit negotiations, changes in the UK's trade relations, or economic policy shifts within the EU and with China.

10.	**Britain and WTI**:
   -  The UK's dependency on oil imports and the influence of global oil prices on the British economy can lead to noticeable spillovers between these two entities.

11.	**Italy and China, Italy and Germany, Italy and WTI**: 
   - Italy's interactions with these markets could be affected by its economic structure, reliance on external trade, and how global market movements impact its domestic financial stability.

12.	**China and Germany, China and WTI**: 
   - China's relationship with Germany reflects broader trade and investment ties, while its relationship with WTI prices shows the impact of energy consumption patterns and economic growth on commodity volatility.

13.	**Germany and WTI**: 
   - Germany, as an industrial powerhouse, has a complex relationship with oil prices that can affect its export-oriented economy.

14.	**WTI with all**: 
   - WTI's interaction with each country's market can be viewed through the lens of oil's fundamental role in the global economy and its impact on energy-dependent sectors and inflation.

Each pairwise interaction can be influenced by numerous factors, including economic policies, trade relations, monetary policy shifts, geopolitical events, and other macroeconomic factors. Specific spikes or downturns would need to be correlated with historical events for precise analysis.

"""
                st.write(text)                


        elif energy_option == 'Gas':
            figures = ['Total Volatility Spillovers',"Directional Volatility Spillovers(To all others)",
                         "Directional Volatility Spillovers(From all others)","Net Volatility Spillovers",
                         "Net Pairwise Volatility Spillovers"]
            figure_option = st.selectbox('Please select a pattern',figures)
            if figure_option=="Total Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/gas_1.png'
                st.image(image_path, caption='Total Volatility Spillovers', use_column_width=True)
                text = """
The chart displays the total volatility spillovers of natural gas futures prices for six countries, spanning from 2018 to 2024. Below is an explanation of the patterns observed in the chart:
1. **Baseline Fluctuations (2018 - Early 2020)**:
   - Before 2020, volatility spillovers were at a moderate level. These baseline fluctuations may reflect the regular supply and demand changes in the natural gas market and seasonal factors.
2. **Early 2020 Volatility Peak**:
   - At the beginning of 2020, there is a significant rise in volatility. This may be related to the early stages of the COVID-19 pandemic, which likely affected natural gas demand and prices due to the slowdown in industrial activities and lockdown measures implemented by various countries.
3. **Gradual Upward Trend from 2020 to 2024**:
   - After 2020, there is a general upward trend in total volatility spillovers. This may reflect the market's uncertainty over the recovery of natural gas demand and concerns over the long-term impact of the energy transition on natural gas demand.
4. **Late-Period Volatility**:
   - The high volatility shown in the latter part of the chart may be related to the global economic recovery, changes in energy market policies (such as carbon emission restrictions and investments in renewable energy), and potential geopolitical events.
"""
                st.write(text)

                         
                
            elif figure_option == "Directional Volatility Spillovers(To all others)":
                image_path = 'system/pictures/risk_analysis/gas_2.png'
                st.image(image_path, caption='Directional Volatility Spillovers(To all others)', use_column_width=True)
                text = """
This graph appears to represent directional volatility spillovers from the United States, Britain, China, France, Italy, and Germany to all other markets in a network over a period from 2018 through 2024. Here's a summary analysis for each:
1. **United States (America TO all others)**:
   - Spikes in volatility spillover are noticeable in 2020, likely due to the impact of the COVID-19 pandemic on global markets. The U.S. as a major economic power has a significant impact on global markets, which is reflected here.
   - Subsequent smaller peaks might be related to later stages of the pandemic, political events, or economic policy changes.
2. **Britain (Britain TO all others)**:
   - The increase in volatility spillovers in 2020 could be related to Brexit uncertainties combined with pandemic-related disruptions.
   - Later volatility indicates continued market sensitivity to British economic and political developments.
3. **China (China TO all others)**:
   - Early 2020 shows a notable increase in volatility spillover, possibly due to the onset of the pandemic, given China's role as the initial epicenter.
   - The following years' trends may reflect China's recovery and the impact of its economic policies on global markets.
4. **France (France TO all others)**:
   - A general increase in volatility spillovers post-2020 might reflect the impact of the pandemic on the French economy and subsequent recovery efforts.
   - The steady rise towards 2024 could indicate France's growing economic influence or market reactions to European fiscal policies.
5. **Italy (Italy TO all others)**:
   - The sharp rise in 2020 corresponds with Italy being severely hit by the COVID-19 pandemic and the resulting economic shock.
   - The decline and then a gradual increase in spillovers could reflect the phases of Italy's economic recovery and the influence of European Central Bank policies.
6. **Germany (Germany TO all others)**:
   - The volatility spillover remains relatively stable until 2020, followed by a sharp increase, possibly due to Germany's strong ties with the global economy and its response to the pandemic.
   - The rise towards 2024 suggests Germany's economic decisions continue to have a significant influence on global markets.
These spillovers illustrate how economic events in one major economy can have ripple effects across global markets. The data points to 2020 as a pivotal year due to the COVID-19 pandemic, with continuing effects in subsequent years, likely due to ongoing economic adjustments and recovery efforts.
"""
                st.write(text)


            elif figure_option == "Directional Volatility Spillovers(From all others)":
                image_path = 'system/pictures/risk_analysis/gas_3.png'
                st.image(image_path, caption='Directional Volatility Spillovers(From all others)', use_column_width=True)

                # The text to display
                text = """
This set of charts seems to show the directional volatility spillovers received by the United States, Britain, 
China, France, Italy, and Germany from all other markets included in the system over time. Here's a breakdown for each country:

1. **United States (America FROM all others)**:
   - Notable peak in 2020, indicating significant volatility received from other markets due to the global impact of the COVID-19 pandemic.
   - More normal levels of volatility reception thereafter, with fluctuations indicating ongoing global economic interactions.

2. **Britain (Britain FROM all others)**:
   - Significant increase in volatility received around 2020, potentially due to Brexit uncertainties coupled with the pandemic.
   - Post-2020, general increase suggests ongoing influence from external economic conditions and possibly Brexit adjustments.

3. **China (China FROM all others)**:
   - Spike in early 2020 reflects the initial impact of the pandemic.
   - Stabilizes but with higher volatility reception compared to previous years, perhaps due to ongoing trade tensions or China's growing influence.

4. **France (France FROM all others)**:
   - Increased reception of volatility in 2020, likely tied to pandemic-related disruptions.
   - Steady rise into 2022 and beyond suggests continued sensitivity to external market movements.

5. **Italy (Italy FROM all others)**:
   - Sharp rise in 2020 likely due to being one of the hardest-hit European countries by the pandemic.
   - Sustained high level of volatility reception indicates ongoing influence from broader economic conditions.

6. **Germany (Germany FROM all others)**:
   - Noticeable increase in volatility received in 2020, possibly due to the pandemic's effects on global supply chains.
   - Fluctuating trend thereafter, with an increase towards the end of the period, suggests responsiveness to external market conditions.

These patterns highlight the interconnectedness of global financial markets and the varying degrees to which economies are affected by external shocks.
"""

                # Use Streamlit to display the text
                st.write(text)

            elif figure_option == "Net Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/gas_4.png'
                st.image(image_path, caption='Net Volatility Spillovers', use_column_width=True)
                text = """
The graph shows net volatility spillovers for the United States, Britain, China, France, Italy, and Germany from 2018 through 2024. These charts indicate the difference between the volatility these countries' markets contribute to others and the volatility they receive from others. Let’s go over the analysis for each country:

1. **United States (NET America)**:
   - The net volatility spillover remains close to zero with a few negative spikes, implying the U.S. market's influence is generally balanced with the influence it receives from other markets. The negative spikes may correspond to specific events where the U.S. market was more affected by external factors.

2. **Britain (NET Britain)**:
   - There is a noticeable drop into negative territory around 2020, which might be associated with Brexit and the initial impact of the COVID-19 pandemic. This suggests that the UK market was significantly influenced by external volatility during this period.

3. **China (NET China)**:
   - China's net spillover is relatively stable with minor fluctuations until a notable drop in 2020, likely due to the economic implications of the pandemic, suggesting an increased influence from other global markets.

4. **France (NET France)**:
   - The net spillover sees an increase in variability post-2020, possibly reflecting the sensitivity of the French market to external economic conditions such as the European debt crisis or changes in EU monetary policy.

5. **Italy (NET Italy)**:
   - Similar to France, Italy experiences increased variability and a trend towards receiving more volatility from other markets, which could be due to its economic challenges and policy responses during the pandemic.

6. **Germany (NET Germany)**:
   - Germany shows an increase in volatility reception in 2020, likely a result of its strong economic ties to global supply chains and sensitivity to changes in global trade and economic conditions.

The analysis of these charts can provide insights into how interconnected the global financial markets are and how external shocks, such as the COVID-19 pandemic, can propagate through the system. It highlights the role of economic events and policy decisions in one country on the stability of financial markets across the globe.
"""
                st.write(text)



            elif figure_option == "Net Pairwise Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/gas_5.png'
                st.image(image_path, caption='Net Pairwise Volatility Spillovers', use_column_width=True)
                text = """
This set of charts shows the net pairwise volatility spillovers between various pairs of entities over a period from roughly 2018 to 2024. Here's an analysis for each pairwise combination:

1. **America-France**:
   - Spillovers fluctuate around the baseline, suggesting a balanced relationship of volatility influence between the two markets with no long-term dominant direction.

2. **America-Britain**:
   - Similar balanced interaction as with France, but with a noticeable dip around 2020 which might reflect heightened uncertainty due to Brexit and COVID-19.

3. **America-Italy**:
   - Volatility spillovers from the U.S. to Italy and vice versa remain generally low, indicating neither market consistently dominates the other in terms of volatility transmission.

4. **America-China**:
   - Significant volatility spillovers around 2020 likely reflect the global market's reaction to the pandemic's initial impact in China.

5. **America-Germany**:
   - A relatively stable relationship with some peaks suggests occasional periods where economic events in one country had a significant impact on the other.

6. **France-Britain**:
   - The post-2020 period shows France receiving more volatility from Britain, possibly due to Brexit-related uncertainty affecting European markets.

7. **France-China**:
   - The volatility spillovers appear relatively subdued except for a few spikes, implying that significant economic events occasionally influence the relationship.

8. **France-Germany**:
   - The relationship is generally stable, with few notable peaks, which could be due to the closely tied economies and coordinated policies within the EU.

9. **Britain-Italy**:
   - An increase in volatility received by Italy from Britain, particularly around the 2020 mark, likely a reflection of both Brexit and the pandemic.

10. **Britain-Germany**:
    - Notable peak around 2020, indicating Britain's economic decisions or events had a significant impact on Germany during this period.

11. **Italy-China**:
    - A low level of volatility spillover suggests limited direct financial interdependence or the balanced influence of the two economies on each other.

12. **Italy-Germany**:
    - Italy appears to be significantly influenced by volatility from Germany around 2020, potentially due to the EU's economic interrelations and COVID-19.

13. **China-Germany**:
    - Notable spikes suggest that key events in China can occasionally impact German market volatility, potentially reflecting trade and investment ties.

These patterns provide insight into the complex web of influences between global financial markets. Periods of heightened spillover often correlate with major economic or geopolitical events that affect markets worldwide.
"""
                st.write(text)



    elif analysis_option == 'National':
        figures = ['Total Volatility Spillovers',"Directional Volatility Spillovers(To all others)",
                         "Directional Volatility Spillovers(From all others)","Net Volatility Spillovers",
                         "Net Pairwise Volatility Spillovers"]
        figure_option = st.selectbox('Please select a pattern',figures)
        if figure_option=="Total Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/usa_1.png'
                st.image(image_path, caption='Total Volatility Spillovers', use_column_width=True)

        elif figure_option == "Directional Volatility Spillovers(To all others)":
                image_path = 'system/pictures/risk_analysis/usa_2.png'
                st.image(image_path, caption='Directional Volatility Spillovers(To all others)', use_column_width=True)

        elif figure_option == "Directional Volatility Spillovers(From all others)":
                image_path = 'system/pictures/risk_analysis/usa_3.png'
                st.image(image_path, caption='Directional Volatility Spillovers(From all others)', use_column_width=True)

        elif figure_option == "Net Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/usa_4.png'
                st.image(image_path, caption='Net Volatility Spillovers', use_column_width=True)

        elif figure_option == "Net Pairwise Volatility Spillovers":
                image_path = 'system/pictures/risk_analysis/usa_5.png'
                st.image(image_path, caption='Net Pairwise Volatility Spillovers', use_column_width=True)
