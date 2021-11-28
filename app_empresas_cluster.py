import pandas as pd
import streamlit as st 
#from deep_translator import GoogleTranslator
from sklearn.cluster import KMeans
#from sklearn.metrics import silhouette_score
import pickle
import os
import random
import yaml
import json
import plotly.express as px

st.set_page_config(layout="wide")

#____________________________________________________________________________________________________________
## LOADING KMEANS MODEL, DATASET, NAME TEXT FILE, HEAD VECTOR TO MODEL
# load the cluster Kmeans model from disk
filename = 'kmeans_model_0.25dataset_s0.30_200clusters.sav'#'kmeans_model_alldataset_s0.38_30clusters.sav'
kmeans = pickle.load(open(os.path.join('./recomendacao',filename), 'rb'))

# load clustered dataset
clusters_agg= pd.read_csv(os.path.join('./recomendacao', 'cluster_empresas_agg_s0.30_200clusters.csv'))

# load names from companies
a_file = open("./recomendacao/data_names.json", "r")
list_names_ = json.load(a_file)

# load sequence of activities to pass through Kmeans model
a_file = open("./recomendacao/activities_cluster.json", "r")
activities_cluster = json.load(a_file)

#load activities that are related to audio, acoustics, sound and/or vibration
acoustic_cnaes = pd.read_csv('./recomendacao/cnaes_acustica.csv')

#load activities with description translated to english
cnaes = pd.read_csv('./recomendacao/translated_CNAES.CSV')
#marking activities in the dataset that are related to audio, acoustics, sound and/or vibration
cnaes.loc[cnaes['cnae_code'].isin(acoustic_cnaes['cnae_code'].tolist()),'acoustic?']='Y'

#______________________________________________________________________________________________________________
#VECTOR STRUCTURE TO ML MODEL
#initial vector with all zeros

dict_activity ={}
for x in activities_cluster:
    dict_activity[x]=0 
#st.write(dict_activity)

#______________________________________________________________________________________________________________
#PAGE STRUCTURE
st.title('Brazilian Market Research')
st.header('Clusters by K-means')

my_expander = st.expander("Search for your market research", expanded=True)
with my_expander:
    #choosing activities
    col1, col2 = st.columns(2)
    activity_desc = col1.selectbox('Choose audio, acoustics or vibration activity', cnaes.loc[cnaes['acoustic?']=='Y','code_description'].tolist())
    #col1.text(activity_desc)


    cnae_options = col2.multiselect(
         'AND Select your other activities',
         cnaes['code_description'].tolist(),
         [x for x in cnaes['code_description'].tolist() if "audio" in x])


    st.write('** **')
    st.write('**OR**')
    col1, col2 = st.columns(2)
    known_company = col1.text_input('Search for known company name')

    
    if known_company=='':
        list_to_choose_names =['No written search']
    else:
        list_to_choose_names = [s for s in list_names_ if known_company.upper() in s]
        
    name_chosen = col2.selectbox('AND Choose name for search', list_to_choose_names)
    

    all_activities = list(set([activity_desc] + cnae_options))

st.write(' ')
st.write(' ')

#INPUT DATA - EITHER ACTIVITIES CHOSEN, OR NAME COMPANY RESERACHED
col1, col2, col3 = st.columns(3)
if col2.button('Click to know more about similar companies!'):

    #option when activities are chosen
    if known_company=='':
        all_activities_code = [x.split(' - ')[0] for x in all_activities]

        #filling dictionary with respective activities to pass to ML model
        for x in all_activities_code:
            if x in dict_activity.keys():
                dict_activity[x]=1
        #activities vector to predict
        company_topredict = list(dict_activity.values())
        #cluster predicted
        y_pred = kmeans.predict([company_topredict])
        #st.dataframe(resultado_cluster[resultado_cluster['cluster']==y_pred[0]])
        
        #information about cluster that activities group belongs to
        selected_cluster = clusters_agg[clusters_agg['cluster']==y_pred[0]]
    else:
    #option when company name is researched
        for x in range(len(clusters_agg)):
            if name_chosen in clusters_agg.loc[x,'lista_common_names']:
                #information about cluster that company name belongs to
                selected_cluster = clusters_agg[clusters_agg['cluster']==x]
                
    st.write(" ")
    st.write(" ")
    
#OUTPUT DATA - INFORMATION ABOUT CLUSTER

    #Info in the cards
    
    #quantitative of companies in this cluster
    n_companies = selected_cluster['cnpj_basico_count'].tolist()[0]   
    #less than one year percentage of companies in this cluster
    pctg_zero = round(100*(selected_cluster['n_empresas_zeroanos_'].tolist()[0])/selected_cluster['cnpj_basico_count'].tolist()[0], 1)
    #median age of a company in this cluster
    median_age = int(selected_cluster['idade_median'].tolist()[0])
    #oldest company age in this cluster
    oldest_age = int(selected_cluster['idade_max'].tolist()[0])
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Nº of companies", str(n_companies))
    col2.metric("% young companies (<1 year)", str(pctg_zero)+'%')
    col3.metric("Median age", str(median_age))
    col4.metric("Eldest company age", str(oldest_age))
    
    #____________________________________________________________
    
    #Information about activity composition and company names
    
    #number of companies for each activity in the cluster
    dict_activ = yaml.full_load(selected_cluster['lista_common_activities'].tolist()[0]) 
    #dataframe with code of activity, description and percentage of companies with that activity
    df_activities = pd.DataFrame(columns = ['cnae','cnae_desc', '%companies'])
    #sorting values with bigger numbers first
    dict_activ_order = sorted(dict_activ.values(), reverse=True)
    
    #calculating 10 most common activities in the cluster and saving it to dataframe
    count=0
    for x in dict_activ_order[:10]:
        for k in dict_activ.keys():
            if (dict_activ[k] == x) and(k!='nan'):
                cnae_desc_name = cnaes.loc[cnaes['cnae_code']==int(k),'code_description'].tolist()[0]
                pctg_activ = round((100*dict_activ[k]/n_companies),1)
                df_activities.loc[count, :] = [cnae_desc_name.split(' - ')[0], cnae_desc_name, pctg_activ]
                count+=1
    #st.dataframe(df_activities)
    
    #activities data to plot
    data_ = df_activities.sort_values(by = '%companies')
    #bar plot - activities composition
    fig = px.bar(data_, y='cnae', x='%companies',
                 hover_data=['cnae_desc', '%companies'], labels={'%companies':'Percentage of companies in this cluster'},
                 title='Activity composition in this cluster', orientation = 'h', text='%companies' #color='%companies',
                 )
                 
    #all company names in this cluster
    dict_names = yaml.full_load(selected_cluster['lista_common_names'].tolist()[0])
    #8 names chosen randomly
    dict_names_random = random.sample(dict_names, 8)
                 
    col1, col2 = st.columns([1.5, 1])
    #displaying bar plot activity composition
    col1.plotly_chart(fig)
    #random company names chosen randomly in the cluster that are not NAN
    col1.write('**Similar companies:**')    
    for x in dict_names_random:
        if x.startswith('NAN'):
            pass
        else:
            col1.write("* "+x)
            
    #displaying only the 10 most common activities in the cluster
    col2.write('**Activities in similar companies:**')    
    for x in range(len(df_activities)):
        col2.write("* **"+str(df_activities.loc[x, '%companies'])+'**% - '+str(df_activities.loc[x, 'cnae_desc']))
   
       
       
#____________________________________________________________________________________________________________
#CREDITS AND INFO ABOUT DATASET

#links to personal and dataset information
#Ingrid
ingrid_photo = 'https://media-exp1.licdn.com/dms/image/C4D03AQH_InG-xiedOA/profile-displayphoto-shrink_800_800/0/1615148707112?e=1642636800&v=beta&t=KVBHsx89CGR_PJGGob87UdRSJD9hsJQbtulcajoIGeg'
url_Ingrid='https://www.linkedin.com/in/ingrid-knochenhauer/'
#Nickolas
nickolas_photo = 'https://media-exp1.licdn.com/dms/image/C4D03AQE7jC72ouPqxw/profile-displayphoto-shrink_400_400/0/1637540142042?e=1643241600&v=beta&t=2HcqcUCQZgrLK08YU4-HTpQb6BQOnqTt4gkZJ2bIPxk'
url_Nickolas='https://www.linkedin.com/in/nickolas-s-mendes/'
#Receita Federal
url_data = 'https://www.gov.br/receitafederal/pt-br/assuntos/orientacao-tributaria/cadastros/consultas/dados-publicos-cnpj'

#PAGE
st.write(' ')
st.write(' ')
st.write("** **")
st.write('This app was developed by:')
col1,col2, _, _ = st.columns([1,1,4,4])
#Ingrid info
col1.image(ingrid_photo)
col1.markdown("[Ingrid](url_Ingrid)")
#Nickolas info
col2.image(nickolas_photo)
col2.markdown("[Nickolas](url_Nickolas)")
#dataset info
st.markdown("**Dataset**: [Companies data](url_data) from 15/10/2021")




# def translation_pt_eng(sentence):
    # translated = GoogleTranslator(source='auto', target='en').translate(sentence) 
    # #st.text(translated)
    # return translated

# cnaes = pd.read_csv('./recomendacao/F.K03200$Z.D11009.CNAECSV', sep=';', encoding='ISO-8859-1', header=None)
# cnaes.columns = ['cnae_code', 'ativ_desc']
# cnaes['description'] = cnaes['ativ_desc'].apply(translation_pt_eng)

# cnaes['code_description'] = cnaes['cnae_code'].astype(str) + ' - ' + cnaes['description']0
