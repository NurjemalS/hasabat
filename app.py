import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# pip install --upgrade numpy pandas

topics = ["Gahryman Arkadagymyzyň öňe süren teklibine laýyklykda geçirilýän maslahatlary geçirmegiň ähmiýetini düzündirmek bilen bagly maslahatlar",
    "XXI asyrda bilimiň mazmunyna we okatmagyň usulyýetine täzeçe garamak, şeýle hem bilim babatda umumy maksatlara ýetmegi çaltlandyrmak boýunça esasy strategik özgertmeleri we gurallary kesgitlemek",
    "Bilim çygrynda milli maksatlary we görkezijileri kesgitlemek",
    "Bilimiň döwlet tarapyndan pugtalandyrylmagyny we has durnukly maliýeleşdirilmegini üpjün etmek",
    "Ýurdumyzda halkara derejesine laýyk gelýän umumybilim edaralaryny döretmek, mekdep-gimnaziýalary açmak",
    "Ylmyň we tehnikanyň örän çalt depginlerde ösmegi, täze tehnologiýalaryň döremegi bilen baglylykda, bilim edaralarynda okatmagyň usulyýetini kämilleşdirmek",
    "Arkadag şäheriniň bilim edaralaryny ÝUNESKO-nyň assosirlenen mekdepler toruna girizmek boýunça zerur işleri alyp barmak",
    "Ýurdumyzda döwrebap bilim edaralaryny gurup, ylmyň we innowasiýalaryň ileri tutulýan ugurlary boýunça tehnologiýalar merkezlerini döretmek",
    "Dünýäniň öňdebaryjy uniwersitetleriniň sanawyna girýän bilelikdäki ýokary okuw mekdeplerini ýa-da olaryň şahamçalaryny döretmek",
    "Mekdebe çenli çagalar edaralarynyň hem-de umumybilim berýän orta mekdepleriň sanyny artdyrmak",
    "Ýokary okuw mekdeplerinde ylym-bilim-önümçilik arabaglanyşygynyň kämil usulyny döretmek",
    "Ýaş nesilleri milli gymmatlyklarymyza, däp-dessurlarymyza buýsanç ruhunda, ynsanperwer kadalarymyz esasynda terbiýelemek üçin meşhur şahsyýetlerimiziň ýadygärliklerini ebedileşdirmek",
    "Müňýyllyklardan gözbaş alýan medeniýetimizi, edebiýatymyzy, şöhratly taryhymyzy täze nazaryýet esasynda düýpli öwrenmek hem-de ylmy taýdan beýan etmek",
    "Arkadag şäheriniň dünýäniň ylym-bilim merkezi hökmündäki ornuny pugtalandyryp, şäherde Türkmenistanyň gadymy, orta asyrlar, täze we iň täze taryhyny, arheologik, binagärlik ýadygärliklerini, döwletimiziň alyp barýan syýasatyny öwrenýän hem-de wagyz edýän halkara ylmy-barlag merkezini döretmek",
    "Ýokary bilimi ösdürmegiň Strategiýasyny taýýarlamak",
    "Intellektual eýeçilik ulgamyny ösdürmegiň Konsepsiýasyny taýýarlamak",
    "Maslahatlaryň jemleýji maslahaty (Beylekiler)"]

# Create a dictionary with topics as keys and ordered numbers as values
topics_map = {topic: idx + 1 for idx, topic in enumerate(topics)}
# print(topics_map)

page_title = "2024-nji ýylyň 24-nji sentýabrynda geçirilen Türkmenistanyň Halk Maslahatynyň mejlisinde türkmen halkynyň Milli Lideri, Türkmenistanyň Halk Maslahatynyň Başlygy Gahryman Arkadagymyzyň, täze döwür amala aşyrylýan özgertmeleri has-da giňeldip, jemgyýetimiziň aň-bilim mümkinçiligini ýokarlandyrmagy talap edýändigi, şoňa görä-de, şu ýylyň ahyryna çenli toplumlarda ylym, bilim, medeniýet ulgamlaryny kämilleşdirmek, ýaşlar barada döwlet syýasaty bilen bagly maslahatlary geçirmek barada öňe süren teklibinden ugur alyp, ýurdumyzyň bilim ulgamynda geçirilen maslahatlar barada MAGLUMAT"

st.set_page_config(page_title=page_title, layout="wide")
st.markdown(
    """
    <style>
    [data-testid="stSidebar"][aria-expanded="true"] > div:first-child{
        width: 300px;
    }
    [data-testid="stSidebar"][aria-expanded="false"] > div:first-child{
        width: 300px;
        margin-left: -400px;
    }
     
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
    .metric-container {
        font-size: 30px !important;  /* Adjust the font size as needed */
        font-weight: bold !important; /* Optional: Make it bold */
    }
    </style>
    """,
    unsafe_allow_html=True
)


# file_path = 'YOM_1.xlsx'
# df_YOM_1 = pd.read_excel(file_path, sheet_name='1-J')


df_YOM_1 = pd.read_csv('YOM_1.csv')
df_YOM_1['age(0-17)'] = pd.to_numeric(df_YOM_1['age(0-17)'], errors='coerce')
df_YOM_1['age(0-17)'].fillna(0, inplace=True)
df_YOM_1['age(0-17)'] = df_YOM_1['age(0-17)'].astype(int)
df_YOM_1['age(0-17)'] = df_YOM_1['age(0-17)'].astype(int)
# print(df_YOM_1.dtypes)
df_OHOM_1 = pd.read_csv('OHOM_1.csv')
df_HTOM_1 = pd.read_csv('HTOM_1.csv')
# print(df_OHOM_1.dtypes)
# print(df_HTOM_1.dtypes)

YOM_participant = df_YOM_1['number_of_participants'].sum()
YOM_meetings = df_YOM_1['number_of_meetings'].sum()
YOM_suggestions = df_YOM_1['number_of_suggestions'].sum()

OHOM_participant = df_OHOM_1['number_of_participants'].sum()
OHOM_meetings = df_OHOM_1['number_of_meetings'].sum()
OHOM_suggestions = df_OHOM_1['number_of_suggestions'].sum()

HTOM_participant = df_HTOM_1['number_of_participants'].sum()
HTOM_meetings = df_HTOM_1['number_of_meetings'].sum()
HTOM_suggestions = df_HTOM_1['number_of_suggestions'].sum()
piecharts = False

# st.write(type(combined_df["age(0-17)"][2]))
# value = combined_df.iloc[2, 13]  # 2nd row (index 1), 4th column (index 3)
# print("Value:", value)
# st.write(type(value))
# print("Type of value:", type(value))


st.sidebar.title("Navigation")

		
page = st.sidebar.radio("Select a category", [
    "Overview",
    "Data Analysis",
    "Suggestions & Topics"
])


if page == "Overview":
    st.header(page_title)

    st.markdown("This page gives an overview of the all collected data.")
    data_type = st.selectbox(
        "Choose a Data Type",
        ["YOM", "OHOM",  "HTOM", "ALL"] )
    
    if data_type == "YOM":
        combined_df = pd.concat([df_YOM_1])
    elif data_type == "OHOM":
        combined_df = pd.concat([df_OHOM_1])
    elif data_type == "HTOM":
        combined_df = pd.concat([df_HTOM_1])
    else:
        piecharts = True
        combined_df = pd.concat([df_YOM_1, df_OHOM_1, df_HTOM_1])


    combined_df["number"] = range(1, len(combined_df) + 1)
    combined_df = combined_df.fillna(0)


    
    st.markdown("<br>", unsafe_allow_html=True)


    # Calculate total participants and total meetings
    total_participants = combined_df['number_of_participants'].sum()
    total_meetings = combined_df['number_of_meetings'].sum()
    total_suggestions = combined_df['number_of_suggestions'].sum()
    # problematic_rows = combined_df[~combined_df["age(0-17)"].str.isnumeric()]
    # st.write(problematic_rows)
    # columns_to_convert = ["age(0-17)"]  
    # combined_df[columns_to_convert] = combined_df[columns_to_convert].astype(int)

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Add content to each column
    with col1:
        st.metric(label="## Total Participants", value=total_participants)
    with col2:
        st.metric(label="## Total Meetings", value=total_meetings)

    with col3:
        st.metric(label="## Total Suggestions", value=total_suggestions)


    participant_percentages = [
        (YOM_participant / total_participants) * 100,
        (OHOM_participant / total_participants) * 100,
        (HTOM_participant / total_participants) * 100,
    ]

    meeting_percentages = [
        (YOM_meetings / total_meetings) * 100,
        (OHOM_meetings / total_meetings) * 100,
        (HTOM_meetings / total_meetings) * 100,
    ]

    suggestion_percentages = [
        (YOM_suggestions / total_suggestions) * 100,
        (OHOM_suggestions / total_suggestions) * 100,
        (HTOM_suggestions / total_suggestions) * 100,
    ]

    # Labels for the datasets
    labels = ["YOM", "OHOM", "HTOM"]

    if piecharts:

    # Streamlit layout
        st.write("### Percentage Contribution of Each Dataset")

        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col1:
            # Pie chart for participants
            st.write("#### Participants Percentage")
            fig1, ax1 = plt.subplots()
            ax1.pie(participant_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["red", "blue", "green"])
            ax1.set_title("Participants Contribution")
            ax1.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.
            st.pyplot(fig1)

        with col2:
            st.write("#### Meetings Percentage")
            fig2, ax2 = plt.subplots()
            ax2.pie(meeting_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["orange", "purple", "cyan"])
            ax2.set_title("Meetings Contribution")
            ax2.axis("equal")
            st.pyplot(fig2)

        with col3:
            st.write("#### Suggestions Percentage")
            fig3, ax3 = plt.subplots()
            ax3.pie(suggestion_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["yellow", "pink", "gray"])
            ax3.set_title("Suggestions Contribution")
            ax3.axis("equal")
            st.pyplot(fig3)

elif page == "Data Analysis":
    data_type = st.selectbox(
        "Choose a Data Type",
        ["YOM", "OHOM",  "HTOM", "ALL"] )
    
    if data_type == "YOM":
        combined_df = pd.concat([df_YOM_1])
    elif data_type == "OHOM":
        combined_df = pd.concat([df_OHOM_1])
    elif data_type == "HTOM":
        combined_df = pd.concat([df_HTOM_1])
    else:
        piecharts = True
        combined_df = pd.concat([df_YOM_1, df_OHOM_1, df_HTOM_1])
    
    combined_df["number"] = range(1, len(combined_df) + 1)
    combined_df = combined_df.fillna(0)

    
    with st.expander("Section: Distribution"):
        st.write("This section contains distribution an overview of the data.")

    # Summary statistics
        st.write("### Summary Statistics")
        st.dataframe(combined_df.describe())
        # st.dataframe(combined_df)

        selected_column = st.selectbox("Choose a Column", ["number_of_meetings", "number_of_participants", "students", "profs", "parents",  "sectors", "age(0-17)", "age(18-29)", "age(30-59)", "60+", "female", "male", "number_of_topics", "number_of_suggestions"])
        distribution_type = st.selectbox(
        "Choose a Distribution Type",
        ["Histogram", "Density Plot",  "Scatterplot"]
        )

        # might change here like col 1 only 

        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col2:
            if distribution_type == "Histogram":
                st.write(f"### Histogram of {selected_column}")
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.hist(combined_df[selected_column], bins=10, color='red', edgecolor='black', alpha=0.7)
                ax.set_title(f"Histogram: {selected_column}")
                ax.set_xlabel(selected_column)
                ax.set_ylabel("Frequency")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

        # Density Plot
            elif distribution_type == "Density Plot":
                st.write(f"### Density Plot of {selected_column}")
                fig, ax = plt.subplots(figsize=(16, 10))
                data = combined_df[selected_column]
                density, bins = np.histogram(data, bins=30, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate center of bins
                ax.plot(bin_centers, density, color='red', linewidth=2)
                ax.fill_between(bin_centers, density, color='red', alpha=0.3)
                ax.set_title(f"Density Plot: {selected_column}")
                ax.set_xlabel(selected_column)
                ax.set_ylabel("Density")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

            # Scatterplot
            elif distribution_type == "Scatterplot":
                x_col = st.selectbox("Select X-axis Column", ["number_of_meetings", "number_of_participants", "students", "profs", "parents",  "sectors", "age(0-17)", "age(18-29)", "age(30-59)", "60+", "female", "male", "number_of_topics", "number_of_suggestions"], index=0)
                y_col = st.selectbox("Select Y-axis Column", ["number_of_meetings", "number_of_participants", "students", "profs", "parents", "sectors","age(0-17)", "age(18-29)", "age(30-59)", "60+", "female", "male", "number_of_topics", "number_of_suggestions"], index=1)
                st.write(f"### Scatterplot: {x_col} vs {y_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    combined_df[x_col], combined_df[y_col],
                    c=combined_df[selected_column], cmap='cool', edgecolor='black', alpha=0.7
                )
                ax.set_title(f"Scatterplot: {x_col} vs {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                fig.colorbar(scatter, label=selected_column)
                ax.grid(linestyle='--', alpha=0.7)
                st.pyplot(fig)

    with st.expander("Section : Participant analysis"):
        col1, col2 = st.columns(2)
        with col1:
            # Participants by Role
            roles = ['students', 'profs', 'parents', 'sectors']
            role_totals = combined_df[roles].sum()
            st.write("### Total Participants by Role")
            st.bar_chart(role_totals)

        with col2:
            # Participants by Age Group
            age_groups = ['age(0-17)', 'age(18-29)', 'age(30-59)', '60+']
            age_totals = combined_df[age_groups].replace('­', 0).replace('-', 0).astype(int).sum()
            st.write("### Total Participants by Age Group")
            st.bar_chart(age_totals)

        # Gender Distribution
        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col1:
            # st.metric(label="Total Participants", value=total_participants)

            gender_totals = combined_df[['female', 'male']].sum()
            st.write("### Gender Distribution")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(gender_totals, labels=['Female', 'Male'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            # st.metric(label="Total Participants", value=total_participants)
            role_totals = combined_df[['students', 'profs', 'parents', 'sectors']].sum()
            st.write("### Role Distribution")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(role_totals, labels=['students', 'profs', 'parents', 'sectors'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
        with col3:
            # st.metric(label="Total Participants", value=total_participants)
            age_totals = combined_df[['age(0-17)', 'age(18-29)', 'age(30-59)', '60+']].sum()
            st.write("### Role Distribution")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(age_totals, labels=['age(0-17)', 'age(18-29)', 'age(30-59)', '60+'], autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)


    with st.expander("Section : Bottom and Top analysis"):
        selected_column = st.selectbox("Choose a Column to Rank By", ["number_of_meetings", "number_of_participants", "students", "profs", "parents", "sectors", "age(0-17)", "age(18-29)", "age(30-59)", "60+", "female", "male", "number_of_topics", "number_of_suggestions"])

    # Get the Top 5 based on the selected column
        top_5 = combined_df.nlargest(5, selected_column)[["name", selected_column]]
        col1, col2 = st.columns(2)
        with col1:
            # Plotting
            st.write(f"### Top 5 by {selected_column.capitalize()}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_5["name"], top_5[selected_column])
            ax.set_title(f"Top 5 by {selected_column.capitalize()}")
            ax.set_ylabel(selected_column.capitalize())
            ax.set_xlabel("Name")
            ax.set_xticklabels(top_5["name"], rotation=45, ha="right")
            st.pyplot(fig)
        with col2:
        # Get Bottom 5 based on the selected column
            bottom_5 = combined_df.nsmallest(5, selected_column)[["name", selected_column]]
            # Plotting
            st.write(f"### Bottom 5 by {selected_column.capitalize()}")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(bottom_5["name"], bottom_5[selected_column], color="orange")  # Use a different color for distinction
            ax.set_title(f"Bottom 5 by {selected_column.capitalize()}")
            ax.set_ylabel(selected_column.capitalize())
            ax.set_xlabel("Name")
            ax.set_xticklabels(bottom_5["name"], rotation=45, ha="right")
            st.pyplot(fig)

    with st.expander("Section: Topics analysis"):
        
        topics = ["topic-1", "topic-2", "topic-3", "topic-4", "topic-5", "topic-6", "topic-7", "topic-8", 
          "topic-9", "topic-10", "topic-11", "topic-12", "topic-13", "topic-14", "topic-15", 
          "topic-16", "topic-17"]        
        for i in range(len(topics)):
            combined_df[topics[i]] = combined_df[topics[i]] * combined_df['number_of_meetings']
        # st.dataframe(combined_df)

        
        suggestions = ["topic-1-suggestion-number", "topic-2-suggestion-number", "topic-3-suggestion-number", "topic-4-suggestion-number", "topic-5-suggestion-number", "topic-6-suggestion-number", "topic-7-suggestion-number", "topic-8-suggestion-number", 
          "topic-9-suggestion-number", "topic-10-suggestion-number", "topic-11-suggestion-number", "topic-12-suggestion-number", "topic-13-suggestion-number", "topic-14-suggestion-number", "topic-15-suggestion-number", 
          "topic-16-suggestion-number", "topic-17-suggestion-number"]

        # Example layers of data (representing different categories, e.g., discussions, suggestions)
        layer1 = []
        layer2 = []
        for col in topics + suggestions:
            combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        for i in topics:
            layer1.append(combined_df[i].sum())
        for i in suggestions:
            layer2.append(combined_df[i].sum())
        
        data_layers = [layer1, layer2]
        colors = ["blue", "orange"]
        labels = ["how many times discussed", "number of suggestions"]

        # print(layer1)
        # print(layer2)

        # Define angles for each topic
        angles = np.linspace(0, 2 * np.pi, len(topics), endpoint=False).tolist()
        angles += angles[:1]
        for layer in data_layers:
            layer.append(layer[0])

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
        for idx, layer in enumerate(data_layers):
            ax.bar(
                angles, 
                layer, 
                color=colors[idx], 
                alpha=0.6, 
                width=0.35,  # Adjust the width for layering
                label=labels[idx]
            )
        
        # Add labels for topics
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(topics, fontsize=8, rotation=45)

        # Title and Legend
        ax.set_title("Topic Discussions and Suggestions Analysis", va='bottom', fontsize=14)
        ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))

        # Streamlit Display
        st.pyplot(fig)

        # st.write(combined_df["topic-6"].sum())
        # st.write(combined_df["topic-6-suggestion-number"].sum())

                # Example data for suggestions and their counts
        # suggestions = [
        #     "Suggestion 1", "Suggestion 2", "Suggestion 3", "Suggestion 4",
        #     "Suggestion 5", "Suggestion 6", "Suggestion 7", "Suggestion 8",
        #     "Suggestion 9", "Suggestion 10", "Suggestion 11", "Suggestion 12",
        #     "Suggestion 13", "Suggestion 14", "Suggestion 15", "Suggestion 16", "Suggestion 17"
        # ]
        # counts = [380, 250, 300, 200, 400, 280, 350, 100, 450, 320, 200, 180, 370, 240, 260, 300, 220]

        # Calculate percentages
        total = sum(layer2)
        percentages = [(count / total) * 100 for count in layer2]

        # Sort data by percentage (optional for better visualization)
        sorted_data = sorted(zip(suggestions, percentages), key=lambda x: x[1], reverse=True)
        suggestions, percentages = zip(*sorted_data)

        # Plot configuration
        fig, ax = plt.subplots(figsize=(10, 8))

        bars = ax.barh(suggestions, percentages, color=plt.cm.tab20.colors)

        # Add percentage labels to bars
        for bar, percentage in zip(bars, percentages):
            ax.text(
                bar.get_width() + 1,  # Position to the right of the bar
                bar.get_y() + bar.get_height() / 2,  # Vertically centered
                f"{percentage:.1f}%",  # Format percentage
                va="center", fontsize=10
            )

        # Chart labels and title
        ax.set_xlabel("Percentage (%)", fontsize=12)
        ax.set_title("Percentage of Suggestions per topic ", fontsize=14, weight="bold")
        ax.invert_yaxis()  # Reverse the order of suggestions for a top-to-bottom view

        # Streamlit Display
        st.pyplot(fig)

        # print(combined_df.dtypes)


elif page == "Suggestions & Topics":

    df_YOM_S = pd.read_csv('FINAL_YOM.csv')
    df_OHOM_S = pd.read_csv('FINAL_OHOM.csv')
    df_HTOM_S = pd.read_csv('FINAL_HTOM.csv')
    
    data_type = st.selectbox(
        "Choose a Data Type",
        ["YOM", "OHOM",  "HTOM"] )
    
    if data_type == "YOM":
        combined_df = pd.concat([df_YOM_S])
    elif data_type == "OHOM":
        combined_df = pd.concat([df_OHOM_S])
    elif data_type == "HTOM":
        combined_df = pd.concat([df_HTOM_S])
    # elif  data_type == "ALL":
    #     combined_df = pd.concat([df_YOM_S, df_OHOM_S, df_HTOM_S])
    
    combined_df.fillna("YOK", inplace=True)

    # st.dataframe(combined_df)

    options = st.multiselect(
        "Select Topics", topics)
    topics_selected = []

    if options:
        for value in options:
            topics_selected.append('topic-' + str(topics_map[value]))


    # print these columns of suggestions
    topics_selected_suggestions = []
    for i in range(len(topics_selected)):
        topics_selected_suggestions.append(topics_selected[i] + "-suggestions")

    suggestions_list = []

    suggestions_list_map = combined_df[topics_selected_suggestions].to_dict()

    for topic, suggestions in suggestions_list_map.items():
        for key, value in suggestions.items():
            if value != "YOK" and value != '-': 
                suggestions_list.append(value)
                
    print(suggestions_list)
    st.code("\n".join(suggestions_list), language="plaintext")


    # choose data set:
    #     multiselect topics and multiselect sectors according to data selected 



    # # # Simple visual summary
    # # st.bar_chart(data=...)  # Example for quick insights
    # # st.markdown("Navigate using the sidebar to explore detailed analyses.")



  

# df_2 = pd.read_excel(file_path, sheet_name='2-J')
# df.columns = ['number', 'abbr', 'name', 'number_of_suggestions', '1-topic-number', '1-topic-suggestion', '2-topic-number', '2-topic-suggestion', '3-topic-number', '3-topic-suggestion', '4-topic-number', '4-topic-suggestion', '5-topic-number', '5-topic-suggestion', '6-topic-number', '6-topic-suggestion', '7-topic-number', '7-topic-suggestion', '8-topic-number', '8-topic-suggestion', '9-topic-number', '9-topic-suggestion', '10-topic-number', '10-topic-suggestion', '11-topic-number', '11-topic-suggestion', '12-topic-number', '12-topic-suggestion', '13-topic-number', '13-topic-suggestion', '14-topic-number', '14-topic-suggestion', '15-topic-number', '15-topic-suggestion', '16-topic-number', '16-topic-suggestion', '17-topic-number', '17-topic-suggestion']
# print(df.head()) 