import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from altair import Chart


st.sidebar.title("IPL Data Analyzer and Prediction")

category = st.sidebar.radio(
    "Choose your option?",
    ('Bowler Analysis', 'Batsman Analysis','Match Analysis' ,'Win Prediction'))

if category == 'Bowler Analysis':
    st.sidebar.write('You selected Bowler Analysis.')
elif category == 'Batsman Analysis':
    st.sidebar.write("You selected Batsman Analysis")
elif category == 'Match Analysis':
    st.sidebar.write("You selected Match Analysis")
else:
    st.sidebar.write("You selected Win Prediction")

batsman_matrix=pd.read_csv('batsman_final_matrix.csv')
batsman_graphs=pd.read_csv('graph_data.csv')
bowler_metrics=pd.read_csv('bowler_metrics_final.csv')
bowler_graphs=pd.read_csv('bowler_final_graph.csv')
batsman=batsman_matrix[['batsman','finish_ability','consistency','running_btw_wicket','hard_hit_ability']]
seasons=batsman_graphs['season'].unique().tolist()
seasons.sort()
toss_season=pd.read_csv('toss_based_win.csv')
max_season_score=pd.read_csv('season_matches_and_score.csv')
team_toss_stats=pd.read_csv('team_toss_stats.csv')


load = st.sidebar.button("Click Here")

if "load_state" not in st.session_state:
    st.session_state.load_state = False


if load or st.session_state.load_state:
    st.session_state.load_state = True
    if category == 'Bowler Analysis':


        # Bowler analysis code goes here

            st.title("Bowler Analysis")

            bowler = bowler_metrics[
                ['bowler', 'economy', 'wic_tak_ability', 'concistency', 'num_times_critical_wicket_taken']]
            sort_by = st.radio(
                "Sort By",
                ('Economy', 'Wic_tak_ability', 'Concistency', 'Critical_wickets'), horizontal=True)

            if sort_by == 'Economy':
                bowler1 = bowler.sort_values(by='economy').reset_index(drop=True)
                bowler1.index = np.arange(1, len(bowler1) + 1)
                st.dataframe(bowler1)
            elif sort_by == 'Wic_tak_ability':
                bowler2 = bowler.sort_values(by='wic_tak_ability').reset_index(drop=True)
                bowler2.index = np.arange(1, len(bowler2) + 1)
                st.dataframe(bowler2)
            elif sort_by == 'Concistency':
                bowler3 = bowler.sort_values(by='concistency').reset_index(drop=True)
                bowler3.index = np.arange(1, len(bowler3) + 1)
                st.dataframe(bowler3)
            else:
                bowler4 = bowler.sort_values(by='num_times_critical_wicket_taken', ascending=False).reset_index(
                    drop=True)
                bowler4.index = np.arange(1, len(bowler4) + 1)
                st.dataframe(bowler4)

            bowler_names=bowler_metrics['bowler'].unique().tolist()
            year_df, names_options = bowler_graphs['season'], bowler_names
            names = st.multiselect("Select Bowler Names", names_options, ["Harbhajan Singh"])
            col1, col2 = st.columns(2)
            with col1:
                st.header('Economy Variation')
                fig, ax = plt.subplots(figsize=(8, 6))
                for bowler in names:
                    x = bowler_graphs[bowler_graphs['bowler'] == bowler]['season']
                    y = bowler_graphs[bowler_graphs['bowler'] == bowler]['economy']
                    ax.plot(x, y, label=bowler)

                # Set the x-axis label, y-axis label, and title
                ax.set_xlabel('Season')
                ax.set_ylabel('Economy')
                ax.set_title('Batsman Economy per Season in IPL')

                # Add a legend
                ax.legend()

                # Display the graph
                st.pyplot(fig)

            with col2:
                st.header('Consistency')
                fig, ax = plt.subplots(figsize=(8, 6))
                for bowler in names:
                    x = bowler_graphs[bowler_graphs['bowler'] == bowler]['season']
                    y = bowler_graphs[bowler_graphs['bowler'] == bowler]['concistency']
                    ax.plot(x, y, label=bowler)

                # Set the x-axis label, y-axis label, and title
                ax.set_xlabel('Season')
                ax.set_ylabel('Consistency')
                ax.set_title('Batsman consistency per Season in IPL')

                # Add a legend
                ax.legend()
                st.pyplot(fig)

                # Display the graph

            col1, col2 = st.columns(2)
            with col1:

                st.header('Yearly Wicket Taken')
                filtered_data = bowler_graphs[bowler_graphs['bowler'].isin(names)]
                colors = sns.color_palette("tab20", 12)
                plot = sns.catplot(x='bowler', y='wic_taken', hue='season', kind='bar', data=filtered_data,
                                   palette=colors)

                # Set the plot title and axis labels
                plt.title('Number of wickets by bowler')
                plt.xlabel('Bowler')
                plt.ylabel('Count')

                # Rotate the x-axis labels for better visibility
                plt.xticks(rotation=45)

                # Show the plot
                st.pyplot(plot)

            with col2:

                st.header('Yearly Critical Wicket')
                colors = sns.color_palette("tab20", 12)
                plot = sns.catplot(x='bowler', y='critical_wicket_taken', hue='season', kind='bar', data=filtered_data,
                                   palette=colors)

                # Set the plot title and axis labels
                plt.title('Number of wickets by bowler')
                plt.xlabel('Bowler')
                plt.ylabel('Count')

                # Rotate the x-axis labels for better visibility
                plt.xticks(rotation=45)

                # Show the plot
                st.pyplot(plot)
    elif category == 'Batsman Analysis':
        st.title("Batsman Analysis")

       # st.title('Sort by')
        category = st.radio(
            "Sort By",
            ('finishing_ability', 'consistency', 'running_btw_wicket', 'hard_hit_ability'),horizontal=True)

        if category == 'finishing_ability':
            st.write('You selected finishing Ability')
            batsman1=batsman.sort_values(by='finish_ability',ascending=False).reset_index(drop=True)
            batsman1.index = np.arange(1, len(batsman1) + 1)
            st.dataframe(batsman1)


        elif category == 'consistency':
            st.write("You selected consistency")
            batsman2 = batsman.sort_values(by='consistency',ascending=False).reset_index(drop=True)
            batsman2.index = np.arange(1, len(batsman2) + 1)
            st.dataframe(batsman2)

        elif category == 'running_btw_wicket':
            st.write("You selected running btw wicket")
            batsman3 = batsman.sort_values(by='running_btw_wicket',ascending=False).reset_index(drop=True)
            batsman3.index = np.arange(1, len(batsman3) + 1)
            st.dataframe(batsman3)

        else:
            st.write("Hard hitting ability")
            batsman4 = batsman.sort_values(by='hard_hit_ability',ascending=False).reset_index(drop=True)
            batsman4.index = np.arange(1, len(batsman4) + 1)
            st.dataframe(batsman4)

        batsman_names = batsman_graphs['batsman'].unique().tolist()
        year_df, names_options = batsman_graphs['season'], batsman_names
        names = st.multiselect("Select Bowler Names", names_options, ["V Kohli"])
        st.subheader('Batsman runs variation')
        fig, ax = plt.subplots(figsize=(8,2.6 ))
        for batsman in names:
            x = batsman_graphs[batsman_graphs['batsman'] == batsman]['season']
            y = batsman_graphs[batsman_graphs['batsman'] == batsman]['batsman_runs']
            ax.plot(x, y, label=batsman)

        # Set the x-axis label, y-axis label, and title
        ax.set_xlabel('Season')
        ax.set_ylabel('Batsman Runs')
        ax.set_title('Batsman runs per Season in IPL')

        # Add a legend
        ax.legend()

        # Display the graph
        st.pyplot(fig)

        st.subheader('Batsman strike rate variation')
        fig, ax = plt.subplots(figsize=(8,2.6 ))
        for batsman in names:
            x = batsman_graphs[batsman_graphs['batsman'] == batsman]['season']
            y = batsman_graphs[batsman_graphs['batsman'] == batsman]['strike_rate']
            ax.plot(x, y, label=batsman)

        # Set the x-axis label, y-axis label, and title
        ax.set_xlabel('Season')
        ax.set_ylabel('Batsman Runs')
        ax.set_title('Batsman Strike rate per Season in IPL')

        # Add a legend
        ax.legend()

        # Display the graph
        st.pyplot(fig)

        filtered_df=batsman_graphs[batsman_graphs['batsman'].isin(names)]
        filtered_df=filtered_df.sort_values(by=['batsman','season'])
        fig = px.scatter(filtered_df, x='batsman_runs', y='strike_rate', size='batsman_runs',
                         color='batsman', hover_name='batsman', animation_frame='season',
                         range_x=[0, 1000], range_y=[0, 250])
        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 2000
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 2
        st.write(fig)

        count_boundary=batsman_matrix[['batsman','no_4','no_6']]
        count_boundary = count_boundary[count_boundary['batsman'].isin(names)]


        # Melt the dataframe to get the counts in a single column
        count_boundary = count_boundary.melt(id_vars=['batsman'], var_name='Boundary', value_name='Count')
        # Plot the count plot using seaborn
        plot=sns.catplot(x='batsman', y='Count', hue='Boundary', kind='bar', data=count_boundary)
        # Set the plot title and axis labels
        plt.title('Number of 4s and 6s for each batsman')
        plt.xlabel('Boundary')
        plt.ylabel('Count')
        plt.xticks(rotation=45)

        # Show the plot
        st.pyplot(plot)


    elif category == 'Match Analysis':
        orange_cap=pd.read_csv('orange_cap.csv')
        purple_cap=pd.read_csv('purple_cap.csv')
        boundaries=pd.read_csv('boundaries.csv')



        st.title('Match Analysis')
        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Select Season")
            season = st.selectbox("Select season", seasons)


        with col2:
            st.header("Orange Cap")
            orange_cap_holder=orange_cap[orange_cap['season']==season]['batsman'].tolist()
            orange_runs=orange_cap[orange_cap['season'] == season]['batsman_runs'].tolist()

            st.subheader(orange_cap_holder[0])
            st.subheader(orange_runs[0])


        with col3:
            st.header("Purple Cap")

            purple_cap_holder = purple_cap[purple_cap['season'] == season]['bowler'].tolist()
            purple_wickets = purple_cap[purple_cap['season'] == season]['player_dismissed'].tolist()

            st.subheader(purple_cap_holder[0])
            st.subheader(purple_wickets[0])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.header("Tournament 6's")

            sixes = boundaries[boundaries['season'] == season]['sixes'].tolist()
            st.subheader(sixes[0])

        with col2:
            st.header("Tournamanent Winner")
            sesson_winner=pd.read_csv('season_winner.csv')
            sesson_winner=sesson_winner[sesson_winner['Year']==season].reset_index()
            st.subheader(sesson_winner['IPL Winner Team'][0])


        with col3:
            st.header("Tournament 4's")
            fours = boundaries[boundaries['season'] == season]['fours'].tolist()
            st.subheader(fours[0])

        yearly=toss_season[toss_season['season']==season]

        yearly = yearly[['winner', 'field_win', 'bat_win']]

        # Melt the dataframe to get the counts in a single column
        yearly = yearly.melt(id_vars=['winner'], var_name='toss_decision', value_name='Count')

        # Plot the count plot using seaborn
        sns.set(style='darkgrid')
        plot=sns.catplot(x='Count', y='winner', hue='toss_decision', kind='bar', data=yearly,sharey=False)
       # plot.set_ylim(0, 80)




        # Set the plot title and axis labels
        plt.title('Match Win based on Toss Decision')
        plt.xlabel('Count')
        plt.ylabel('Teams')
        st.pyplot(plot)



        toss_based_percent_winning = toss_season.groupby(['season'])['field_win', 'bat_win'].sum().reset_index()
        toss_based_percent_winning['field_win_percent'] = toss_based_percent_winning['field_win'] * 100 / (
                    toss_based_percent_winning['field_win'] + toss_based_percent_winning['bat_win'])
        toss_based_percent_winning['field_win_percent'] = toss_based_percent_winning['field_win_percent'].round(
            decimals=2)
        toss_based_percent_winning['bat_win_percent'] = 100 - toss_based_percent_winning['field_win_percent']
        toss_based_percent_winning_yearly=toss_based_percent_winning[toss_based_percent_winning['season']==season].reset_index()

        field_win_percent=toss_based_percent_winning_yearly['field_win_percent'][0]
        bat_win_percent = toss_based_percent_winning_yearly['bat_win_percent'][0]

        sizes=[field_win_percent,bat_win_percent]
        labels = ["Filed Win", "Bat Win"]

        fig,ax=plt.subplots()
        ax.pie(sizes,labels=labels,autopct='%1.1f%%',startangle=90)
        st.pyplot(fig)

        import matplotlib.pyplot as plt
        import pandas as pd

        # create a dataframe with the data


        # create a figure with two subplots
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # create the bar plot
        ax1.bar(max_season_score['season'], max_season_score['matches_played'])
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Number of Matches Played')
        ax1.set_title('Matches Played per Season')

        # create the scatter plot
        ax1.scatter(max_season_score['season'], max_season_score['highest_score'], marker='s')
        ax1.set_xlabel('Season')
        ax1.set_ylabel('Maximum Score')
        ax1.set_title('Highest Score per Season')

        # display the plot
        st.pyplot(fig)

        ##Most successfull IPL team

        most_successfull=team_toss_stats[['Team','matches_win']]
        most_successfull.sort_values(by='matches_win',ascending=False,inplace=True)
        sns.set_style('whitegrid')
        

        fig, ax = plt.subplots()

        # Plot the seaborn bar plot on the created axes
        sns.barplot(y=most_successfull.Team, x=most_successfull.matches_win, ax=ax)

        # Set plot limits and title
        ax.set_xlim(0, 110)
        ax.set_title('Number of matches won by Team')

        # Display the plot in Streamlit
        st.pyplot(fig)

        ## no of matches according to venue
        matches=pd.read_csv('matches.csv')


        # Create a figure and an axes with the desired size
        fig, ax = plt.subplots(figsize=(20, 20))

        # Plot the seaborn count plot on the created axes
        sns.countplot(y='venue', data=matches, ax=ax)

        # Display the plot in Streamlit
        st.pyplot(fig)

        # percentage of match winning after toss winning
        team_names = team_toss_stats['Team'].unique().tolist()

        team= st.selectbox("Select Team Name", team_names)
        st.subheader('Percentage of match winning after toss winning')
        df=team_toss_stats[team_toss_stats['Team']==team].reset_index()
        match_win_bat_per=df['match_win_bat'][0]*100/df['toss_win_count'][0]
        match_win_field_per=df['match_win_field'][0]*100/df['toss_win_count'][0]



        st.write(match_win_bat_per)


        s = [match_win_bat_per, match_win_field_per]
        labels = ["Filed Win", "Bat Win"]


        fig, ax = plt.subplots()
        ax.pie(s, labels=labels, autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)











    else:
        st.title("IPL Win Predictor")
        # Prediction code goes here

        teams=['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

        cities=['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

        pipe = pd.read_pickle(r'new_pickle.pkl')
        col1,col2=st.columns(2)

        with col1:
            batting_team=st.selectbox('Select the batting team',sorted(teams))
        with col2:
            bowling_team=st.selectbox('select the bowling team',sorted(teams))

        selected_city=st.selectbox('Select host city',sorted(cities))

        target=st.number_input('Target')

        col3,col4,col5=st.columns(3)

        with col3:
            score=st.number_input('Score')
        with col4:
            overs=st.number_input('Overs completed')
        with col5:
            wickets=st.number_input('Wickets out')

        if st.button('Predict Probability'):
            runs_left=target-score
            balls_left=120-(overs*6)
            wickets=10-wickets
            crr=score/overs
            rrr=runs_left*6/balls_left


            input_df = pd.DataFrame({'batting_team':[batting_team],'bowling_team':[bowling_team],'city':[selected_city],'runs_left':[runs_left],'balls_left':[balls_left],'wickets':[wickets],'total_runs_x':[target],'crr':[crr],'rrr':[rrr]})
            result = pipe.predict_proba(input_df)
            loss = result[0][0]
            win = result[0][1]
            st.header(batting_team + "- " + str(round(win * 100)) + "%")
            st.header(bowling_team + "- " + str(round(loss * 100)) + "%")






