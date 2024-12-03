import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import plotly.express as px

# pip install --upgrade numpy pandas
# number,abbr,name,number_of_meetings,number_of_participants,students,profs,parents,sectors,age(0-17),age(18-29),age(30-59),60+,female,male,number_of_topics,number_of_suggestions,topic-1,topic-2,topic-3,topic-4,topic-5,topic-6,topic-7,topic-8,topic-9,topic-10,topic-11,topic-12,topic-13,topic-14,topic-15,topic-16,topic-17,topic-1-suggestion-number,topic-2-suggestion-number,topic-3-suggestion-number,topic-4-suggestion-number,topic-5-suggestion-number,topic-6-suggestion-number,topic-7-suggestion-number,topic-8-suggestion-number,topic-9-suggestion-number,topic-10-suggestion-number,topic-11-suggestion-number,topic-12-suggestion-number,topic-13-suggestion-number,topic-14-suggestion-number,topic-15-suggestion-number,topic-16-suggestion-number,topic-17-suggestion-number
# number,abbr,ady,ýygnaklaryň_jemi,gatnaşyjylaryň_jemi,talyplar,mugallymlar,ene-atalar,pudak_edaralar,ýaş(0-17),ýaş(18-29),ýaş(30-59),60+,zenan,erkek,tema_sany,teklip_sany,tema-1,tema-2,tema-3,tema-4,tema-5,tema-6,tema-7,tema-8,tema-9,tema-10,tema-11,tema-12,tema-13,tema-14,tema-15,tema-16,tema-17,tema-1-teklip-sany,tema-2-teklip-sany,tema-3-teklip-sany,tema-4-teklip-sany,tema-5-teklip-sany,tema-6-teklip-sany,tema-7-teklip-sany,tema-8-teklip-sany,tema-9-teklip-sany,tema-10-teklip-sany,tema-11-teklip-sany,tema-12-teklip-sany,tema-13-teklip-sany,tema-14-teklip-sany,tema-15-teklip-sany,tema-16-teklip-sany,tema-17-teklip-sany

# number,abbr,name,total_suggestion,topic-1-suggestions,topic-2-suggestions,topic-3-suggestions,topic-4-suggestions,topic-5-suggestions,topic-6-suggestions,topic-7-suggestions,topic-8-suggestions,topic-9-suggestions,topic-10-suggestions,topic-11-suggestions,topic-12-suggestions,topic-13-suggestions,topic-14-suggestions,topic-15-suggestions,topic-16-suggestions,topic-17-suggestions


topics = ["1. Gahryman Arkadagymyzyň öňe süren teklibine laýyklykda geçirilýän maslahatlary geçirmegiň ähmiýetini düzündirmek bilen bagly maslahatlar",
    "2. XXI asyrda bilimiň mazmunyna we okatmagyň usulyýetine täzeçe garamak, şeýle hem bilim babatda umumy maksatlara ýetmegi çaltlandyrmak boýunça esasy strategik özgertmeleri we gurallary kesgitlemek",
    "3. Bilim çygrynda milli maksatlary we görkezijileri kesgitlemek",
    "4. Bilimiň döwlet tarapyndan pugtalandyrylmagyny we has durnukly maliýeleşdirilmegini üpjün etmek",
    "5. Ýurdumyzda halkara derejesine laýyk gelýän umumybilim edaralaryny döretmek, mekdep-gimnaziýalary açmak",
    "6. Ylmyň we tehnikanyň örän çalt depginlerde ösmegi, täze tehnologiýalaryň döremegi bilen baglylykda, bilim edaralarynda okatmagyň usulyýetini kämilleşdirmek",
    "7. Arkadag şäheriniň bilim edaralaryny ÝUNESKO-nyň assosirlenen mekdepler toruna girizmek boýunça zerur işleri alyp barmak",
    "8. Ýurdumyzda döwrebap bilim edaralaryny gurup, ylmyň we innowasiýalaryň ileri tutulýan ugurlary boýunça tehnologiýalar merkezlerini döretmek",
    "9. Dünýäniň öňdebaryjy uniwersitetleriniň sanawyna girýän bilelikdäki ýokary okuw mekdeplerini ýa-da olaryň şahamçalaryny döretmek",
    "10. Mekdebe çenli çagalar edaralarynyň hem-de umumybilim berýän orta mekdepleriň sanyny artdyrmak",
    "11. Ýokary okuw mekdeplerinde ylym-bilim-önümçilik arabaglanyşygynyň kämil usulyny döretmek",
    "12. Ýaş nesilleri milli gymmatlyklarymyza, däp-dessurlarymyza buýsanç ruhunda, ynsanperwer kadalarymyz esasynda terbiýelemek üçin meşhur şahsyýetlerimiziň ýadygärliklerini ebedileşdirmek",
    "13. Müňýyllyklardan gözbaş alýan medeniýetimizi, edebiýatymyzy, şöhratly taryhymyzy täze nazaryýet esasynda düýpli öwrenmek hem-de ylmy taýdan beýan etmek",
    "14. Arkadag şäheriniň dünýäniň ylym-bilim merkezi hökmündäki ornuny pugtalandyryp, şäherde Türkmenistanyň gadymy, orta asyrlar, täze we iň täze taryhyny, arheologik, binagärlik ýadygärliklerini, döwletimiziň alyp barýan syýasatyny öwrenýän hem-de wagyz edýän halkara ylmy-barlag merkezini döretmek",
    "15. Ýokary bilimi ösdürmegiň Strategiýasyny taýýarlamak",
    "16. Intellektual eýeçilik ulgamyny ösdürmegiň Konsepsiýasyny taýýarlamak",
    "17. Maslahatlaryň jemleýji maslahaty (Beylekiler)"]

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
df_YOM_1['ýaş(0-17)'] = pd.to_numeric(df_YOM_1['ýaş(0-17)'], errors='coerce')
df_YOM_1['ýaş(0-17)'].fillna(0, inplace=True)
df_YOM_1['ýaş(0-17)'] = df_YOM_1['ýaş(0-17)'].astype(int)
df_YOM_1['ýaş(0-17)'] = df_YOM_1['ýaş(0-17)'].astype(int)
# print(df_YOM_1.dtypes)
df_OHOM_1 = pd.read_csv('OHOM_1.csv')
df_HTOM_1 = pd.read_csv('HTOM_1.csv')
# print(df_OHOM_1.dtypes)
# print(df_HTOM_1.dtypes)

YOM_participant = df_YOM_1['gatnaşyjylaryň_jemi'].sum()
YOM_meetings = df_YOM_1['ýygnaklaryň_jemi'].sum()
YOM_suggestions = df_YOM_1['teklip_sany'].sum()

OHOM_participant = df_OHOM_1['gatnaşyjylaryň_jemi'].sum()
OHOM_meetings = df_OHOM_1['ýygnaklaryň_jemi'].sum()
OHOM_suggestions = df_OHOM_1['teklip_sany'].sum()

HTOM_participant = df_HTOM_1['gatnaşyjylaryň_jemi'].sum()
HTOM_meetings = df_HTOM_1['ýygnaklaryň_jemi'].sum()
HTOM_suggestions = df_HTOM_1['teklip_sany'].sum()
piecharts = False

# st.write(type(combined_df["age(0-17)"][2]))
# value = combined_df.iloc[2, 13]  # 2nd row (index 1), 4th column (index 3)
# print("Value:", value)
# st.write(type(value))
# print("Type of value:", type(value))


st.sidebar.title("Nawigasiýa")

		
page = st.sidebar.radio("Kategoriýa saýlaň", [
    "Umumy gözden geçiriş",
    "Maglumat seljerişi",
    "Teklipler we temalar"
])


if page == "Umumy gözden geçiriş":
    st.header(page_title)

    st.markdown("Bu sahypada umumy gözden geçirilen ähli toplanan maglumatlaryň jemi görkezilyär.")
    data_type = st.selectbox(
        "Maglumat saýlaň",
        ["ÝOM", "OHOM",  "HTOM", "Ählisi"] )
    
    if data_type == "ÝOM":
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
    total_participants = combined_df['gatnaşyjylaryň_jemi'].sum()
    total_meetings = combined_df['ýygnaklaryň_jemi'].sum()
    total_suggestions = combined_df['teklip_sany'].sum()
    # problematic_rows = combined_df[~combined_df["age(0-17)"].str.isnumeric()]
    # st.write(problematic_rows)
    # columns_to_convert = ["age(0-17)"]  
    # combined_df[columns_to_convert] = combined_df[columns_to_convert].astype(int)

    # Create three columns
    col1, col2, col3 = st.columns(3)

    # Add content to each column
    with col1:
        st.metric(label="## Maslahata gatnaşyjylaryň jemi ", value=total_participants)
    with col2:
        st.metric(label="## Geçirilen maslahatlaryň jemi", value=total_meetings)

    with col3:
        st.metric(label="## Maslahatyň netijesinde hödürlenen teklipleriň jemi", value=total_suggestions)


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
    labels = ["ÝOM", "OHOM", "HTOM"]

    if piecharts:

    # Streamlit layout
        # st.write("### Percentage Contribution of Each Dataset")
        st.write("### Her maglumat gorunuň göterim goşandy")

        

        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col1:
            # Pie chart for participants
            st.write("#### Maslahata gatnaşyjylaryň göterimi ")
            # st.write("#### Participants Percentage")
            fig1, ax1 = plt.subplots()
            ax1.pie(participant_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#90ee90", "#87cefa", "#f59393"])
            st.write("Maslahata gatnaşyjylaryň goşandy")
            # ax1.set_title("Participants Contribution")
            ax1.axis("equal")  # Equal aspect ratio ensures the pie chart is circular.

            st.pyplot(fig1)

        with col2:
            # st.write("#### Meetings Percentage")
            st.write("#### Geçirilen maslahatlaryň göterimi")
            fig2, ax2 = plt.subplots()
            ax2.pie(meeting_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#90ee90", "#87cefa", "#f59393"])
            # ax2.set_title("Meetings Contribution")
            st.write("Geçirilen maslahatlaryň goşandy")
            ax2.axis("equal")

            st.pyplot(fig2)

        with col3:
            # st.write("#### Suggestions Percentage")
            st.write("#### Teklipleriň göterimi")
            fig3, ax3 = plt.subplots()
            ax3.pie(suggestion_percentages, labels=labels, autopct="%1.1f%%", startangle=90, colors=["#90ee90", "#87cefa", "#f59393"])
            # ax3.set_title("Suggestions Contribution")
            st.write(" Maslahatyň netijesinde hödürlenen teklipleriň goşandy")
            ax3.axis("equal")
            st.pyplot(fig3)

elif page == "Maglumat seljerişi":
    data_type = st.selectbox(
        "Maglumaty saýlaň",
        ["ÝOM", "OHOM",  "HTOM", "Ählisi"] )
    
    if data_type == "ÝOM":
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

    
    with st.expander("Bap: Bölüniş"):
        st.write("Bu bölümde maglumatlaryň bölünişi we umumy görnüşi görkezilýär")

    # Summary statistics
        st.write("### Umumy statistikalar")
        st.dataframe(combined_df.describe())
        # st.dataframe(combined_df)

        cols = ["ýygnaklaryň_jemi","gatnaşyjylaryň_jemi","talyplar","mugallymlar","ene-atalar","pudak_edaralar","ýaş(0-17)","ýaş(18-29)","ýaş(30-59)","60+","zenan","erkek","tema_sany","teklip_sany"]
        selected_column = st.selectbox("Sütün saýlaň", cols)
        distribution_type = st.selectbox(
        # "Choose a Distribution Type",
        "Distribusiýa görnüşini saýlaň",
        ["Histogramma", "Dykyzlyk grafika",  "Sepme diagramma"]
        )

        # might change here like col 1 only 

        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col2:
            if distribution_type == "Histogramma":
                st.write(f"### Histogramma - {selected_column}")
                fig, ax = plt.subplots(figsize=(16, 10))
                ax.hist(combined_df[selected_column], bins=10, color='#90ee90', edgecolor='black', alpha=0.7)
                ax.set_title(f"Histogramma: {selected_column}")
                ax.set_xlabel(selected_column)
                ax.set_ylabel("Gaýtalanmasy")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

        # Density Plot
            elif distribution_type == "Dykyzlyk grafika":
                st.write(f"### Dykyzlyk grafika - {selected_column}")
                fig, ax = plt.subplots(figsize=(16, 10))
                data = combined_df[selected_column]
                density, bins = np.histogram(data, bins=30, density=True)
                bin_centers = 0.5 * (bins[:-1] + bins[1:])  # Calculate center of bins
                ax.plot(bin_centers, density, color='red', linewidth=2)
                ax.fill_between(bin_centers, density, color='red', alpha=0.3)
                ax.set_title(f"Dykyzlyk grafika: {selected_column}")
                ax.set_xlabel(selected_column)
                ax.set_ylabel("Dykyzlyk")
                ax.grid(axis='y', linestyle='--', alpha=0.7)
                st.pyplot(fig)

            # Scatterplot
            elif distribution_type == "Sepme diagramma":
                x_col = st.selectbox("X oky üçin sütüni saýlaň", cols, index=0)
                y_col = st.selectbox("Y oky üçin sütüni saýlaň", cols, index=1)
                st.write(f"### Sepme diagramma: {x_col} - {y_col}")
                fig, ax = plt.subplots(figsize=(10, 6))
                scatter = ax.scatter(
                    combined_df[x_col], combined_df[y_col],
                    c=combined_df[selected_column], cmap='cool', edgecolor='black', alpha=0.7
                )
                ax.set_title(f"Sepme diagramma: {x_col} we {y_col}")
                ax.set_xlabel(x_col)
                ax.set_ylabel(y_col)
                fig.colorbar(scatter, label=selected_column)
                ax.grid(linestyle='--', alpha=0.7)
                st.pyplot(fig)

    with st.expander("Bap: Gatnaşyjylaryň seljermesi"):
        col1, col2 = st.columns(2)
        with col1:
            # Participants by Role
            roles = ['talyplar', 'mugallymlar', 'ene-atalar', 'pudak_edaralar']
            role_totals = combined_df[roles].sum()
            # st.write("### Total Participants by Role")
            st.write("### Hünäri boýunça gatnaşyjylar")
            st.bar_chart(role_totals)

        with col2:
            # Participants by Age Group
            age_groups = ['ýaş(0-17)', 'ýaş(18-29)', 'ýaş(30-59)', '60+']
            age_totals = combined_df[age_groups].replace('­', 0).replace('-', 0).astype(int).sum()
            # st.write("### Total Participants by Age Group")
            st.write("### Ýaşy boýunça gatnaşyjylar")

            st.bar_chart(age_totals)

        # Gender Distribution
        col1, col2, col3 = st.columns(3)

        # Add content to each column
        with col1:
            # st.metric(label="Total Participants", value=total_participants)

            gender_totals = combined_df[['zenan', 'erkek']].sum()
            st.write("### Jynsy boýunça bölünişi")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(gender_totals, labels=['Zenan', 'Erkek'], autopct='%1.1f%%', startangle=90, colors=["#f59393", "#87cefa" ])
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            # st.metric(label="Total Participants", value=total_participants)
            role_totals = combined_df[['talyplar', 'mugallymlar', 'ene-atalar', 'pudak_edaralar']].sum()
            st.write("### Hünäri boýunça bölünişi")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(role_totals, labels=['talyplar', 'mugallymlar', 'ene-atalar', 'pudak_edaralar'], autopct='%1.1f%%', startangle=90, colors=["#90ee90", "#87cefa", "#f59393", "#cb7bed"])
            ax.axis('equal')
            st.pyplot(fig)
        with col3:
            # st.metric(label="Total Participants", value=total_participants)
            age_totals = combined_df[['ýaş(0-17)', 'ýaş(18-29)', 'ýaş(30-59)', '60+']].sum()
            st.write("### Ýaşy boýunça bölünişi")
            fig, ax = plt.subplots()
            fig, ax = plt.subplots(figsize=(6, 4))  # Adjust width and height
            ax.pie(age_totals, labels=['ýaş(0-17)', 'ýaş(18-29)', 'ýaş(30-59)', '60+'], autopct='%1.1f%%', startangle=90, colors=["#f59393", "#90ee90", "#87cefa","#cb7bed"])
            ax.axis('equal')
            st.pyplot(fig)


    with st.expander("Bap: Aşaky we ýokarky seljerişi"):
        selected_column = st.selectbox("Reýting üçin sütüni saýlaň", ["ýygnaklaryň_jemi","gatnaşyjylaryň_jemi","talyplar","mugallymlar","ene-atalar","pudak_edaralar","ýaş(0-17)","ýaş(18-29)","ýaş(30-59)","60+","zenan","erkek","tema_sany","teklip_sany"])

    # Get the Top 5 based on the selected column
        top_5 = combined_df.nlargest(5, selected_column)[["ady", selected_column]]
        col1, col2 = st.columns(2)
        with col1:
            # Plotting
            st.write(f"### {selected_column.capitalize()} boýunça ýokarky görkezijiler ")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(top_5["ady"], top_5[selected_column])
            ax.set_title(f"{selected_column.capitalize()} boýunça ýokarky görkezijiler ")
            ax.set_ylabel(selected_column.capitalize())
            ax.set_xlabel("Ady")
            ax.set_xticklabels(top_5["ady"], rotation=45, ha="right")
            st.pyplot(fig)
        with col2:
        # Get Bottom 5 based on the selected column
            bottom_5 = combined_df.nsmallest(5, selected_column)[["ady", selected_column]]
            # Plotting
            st.write(f"### {selected_column.capitalize()} boýunça pes görkezijiler")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(bottom_5["ady"], bottom_5[selected_column], color="orange")  # Use a different color for distinction
            ax.set_title(f"{selected_column.capitalize()} boýunça pes görkezijiler")
            ax.set_ylabel(selected_column.capitalize())
            ax.set_xlabel("Ady")
            ax.set_xticklabels(bottom_5["ady"], rotation=45, ha="right")
            st.pyplot(fig)

    with st.expander("Bap: Tema seljerişi"):
        
        topics = ["tema-1", "tema-2", "tema-3", "tema-4", "tema-5", "tema-6", "tema-7", "tema-8", 
          "tema-9", "tema-10", "tema-11", "tema-12", "tema-13", "tema-14", "tema-15", 
          "tema-16", "tema-17"]        
        for i in range(len(topics)):
            combined_df[topics[i]] = combined_df[topics[i]] * combined_df['ýygnaklaryň_jemi']
        # st.dataframe(combined_df)


        suggestions = ["tema-1-teklip-sany", "tema-2-teklip-sany", "tema-3-teklip-sany", "tema-4-teklip-sany", "tema-5-teklip-sany", "tema-6-teklip-sany", "tema-7-teklip-sany", "tema-8-teklip-sany", 
          "tema-9-teklip-sany", "tema-10-teklip-sany", "tema-11-teklip-sany", "tema-12-teklip-sany", "tema-13-teklip-sany", "tema-14-teklip-sany", "tema-15-teklip-sany", 
          "tema-16-teklip-sany", "tema-17-teklip-sany"]

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
        labels = ["ara alyp maslahatlaşmalaryň sany", "teklipleriň sany"]

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
        # ax.set_title("Topic Discussions and Suggestions Analysis", va='bottom', fontsize=14)
        ax.set_title("Temalar boýunça ara alyp maslahatlaşmalar we teklipler seljermesi", va='bottom', fontsize=14)
    
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
        ax.set_xlabel("Göterim (%)", fontsize=12)
        ax.set_title("Temalar boýunça teklipleriň göterimi ", fontsize=14, weight="bold")
        ax.invert_yaxis()  # Reverse the order of suggestions for a top-to-bottom view

        # Streamlit Display
        st.pyplot(fig)

              
            # Participants by Role
        # roles = ['students', 'profs', 'parents', 'sectors']
        suggestions_per_topic_numbers = ["tema-1-teklip-sany", "tema-2-teklip-sany", "tema-3-teklip-sany", "tema-4-teklip-sany", "tema-5-teklip-sany", "tema-6-teklip-sany", "tema-7-teklip-sany", "tema-8-teklip-sany", 
          "tema-9-teklip-sany", "tema-10-teklip-sany", "tema-11-teklip-sany", "tema-12-teklip-sany", "tema-13-teklip-sany", "tema-14-teklip-sany", "tema-15-teklip-sany", 
          "tema-16-teklip-sany", "tema-17-teklip-sany"]
        suggestions_totals = combined_df[suggestions_per_topic_numbers].sum()
        print(suggestions_totals.sort_values(ascending=True))
        sorted_suggestions = suggestions_totals.sort_values(ascending=True)
        st.write("### Her tema boýunça teklipleriň sany")
        st.bar_chart(sorted_suggestions)

        # print(combined_df.dtypes)


elif page == "Teklipler we temalar":
    # number,abbr,ady,teklip_sany,tema-1-teklip,tema-2-teklip,tema-3-teklip,tema-4-teklip,tema-5-teklip,tema-6-teklip,tema-7-teklip,tema-8-teklip,tema-9-teklip,tema-10-teklip,tema-11-teklip,tema-12-teklip,tema-13-teklip,tema-14-teklip,tema-15-teklip,tema-16-teklip,tema-17-teklip
    # number,abbr,name,total_suggestion,topic-1-suggestions,topic-2-suggestions,topic-3-suggestions,topic-4-suggestions,topic-5-suggestions,topic-6-suggestions,topic-7-suggestions,topic-8-suggestions,topic-9-suggestions,topic-10-suggestions,topic-11-suggestions,topic-12-suggestions,topic-13-suggestions,topic-14-suggestions,topic-15-suggestions,topic-16-suggestions,topic-17-suggestions

    flagAll = False
    df_YOM_S = pd.read_csv('FINAL_YOM.csv')
    df_OHOM_S = pd.read_csv('FINAL_OHOM.csv')
    df_HTOM_S = pd.read_csv('FINAL_HTOM.csv')
    
    data_type = st.selectbox(
        "Maglumat saýlaň",
        ["ÝOM", "OHOM",  "HTOM", "Ählisi"] )
    
    if data_type == "ÝOM":
        combined_df = pd.concat([df_YOM_S])
    elif data_type == "OHOM":
        combined_df = pd.concat([df_OHOM_S])
    elif data_type == "HTOM":
        combined_df = pd.concat([df_HTOM_S])
    else:
        flagAll = True
        combined_df = pd.concat([df_OHOM_S, df_HTOM_S, df_YOM_S])
    
    # combined_df["number"] = range(1, len(combined_df) + 1)
    combined_df.fillna("YOK", inplace=True)

    # st.dataframe(combined_df)

    options = st.multiselect(
        "Tema saýlaň", topics)
    topics_selected = []

    if options:
        for value in options:
            topics_selected.append('tema-' + str(topics_map[value]))
    
    # st.write(options)
    # st.dataframe(combined_df)


    # print these columns of suggestions
    topics_selected_suggestions = []
    for i in range(len(topics_selected)):
        topics_selected_suggestions.append(topics_selected[i] + "-teklip")
    


    suggestions_list = []
    count = 0
    print(combined_df[topics_selected_suggestions])

    suggestions_list_map = combined_df[topics_selected_suggestions].to_dict()
    print("SUGGESTION MAP")          

    print(suggestions_list_map)

    if not flagAll:
        countI = 0
        for topic, suggestions in suggestions_list_map.items():
            for key, value in suggestions.items():
                if value != "YOK" and value != '-': 
                    countI += 1
                    suggestions_list.append(value)
    
    if flagAll:
        count = 0
        for index, row in combined_df.iterrows():
            for k in topics_selected_suggestions:
                if row[k] != "YOK" and row[k] != '-':
                    count += 1
                    suggestions_list.append(row[k])
                    print(row[k])

    # print(len(suggestions_list))
    # print("COUNT")
    # print(count)

    # print("SUGGESTION LIST")          
    # print(suggestions_list)
    if flagAll:
        st.metric(label="## Saýlanan teklipleriň jemi sany", value=count)
    else:
        st.metric(label="## Saýlanan teklipleriň jemi sany", value=countI)

    
    st.code("\n".join(suggestions_list), language="plaintext")


    # choose data set:
    #     multiselect topics and multiselect sectors according to data selected 


    # # # Simple visual summary
    # # st.bar_chart(data=...)  # Example for quick insights
    # # st.markdown("Navigate using the sidebar to explore detailed analyses.")


# df_2 = pd.read_excel(file_path, sheet_name='2-J')
# df.columns = ['number', 'abbr', 'name', 'number_of_suggestions', '1-topic-number', '1-topic-suggestion', '2-topic-number', '2-topic-suggestion', '3-topic-number', '3-topic-suggestion', '4-topic-number', '4-topic-suggestion', '5-topic-number', '5-topic-suggestion', '6-topic-number', '6-topic-suggestion', '7-topic-number', '7-topic-suggestion', '8-topic-number', '8-topic-suggestion', '9-topic-number', '9-topic-suggestion', '10-topic-number', '10-topic-suggestion', '11-topic-number', '11-topic-suggestion', '12-topic-number', '12-topic-suggestion', '13-topic-number', '13-topic-suggestion', '14-topic-number', '14-topic-suggestion', '15-topic-number', '15-topic-suggestion', '16-topic-number', '16-topic-suggestion', '17-topic-number', '17-topic-suggestion']
# print(df.head()) 
# data type icinde filterleme 

