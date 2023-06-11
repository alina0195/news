import streamlit as st 
import time
import pandas as pd
from wordcloud import WordCloud, STOPWORDS
import re
import pymongo
from pymongo import MongoClient
from matplotlib import pyplot as plt
import torch 
import torch.nn as nn
from transformers import AutoTokenizer
from transformers import AutoTokenizer, TFPegasusForConditionalGeneration, T5ForConditionalGeneration

class Config:
    MAX_TEXT_LEN_CAT = 256
    MAX_LEN_HEADLINE = 512
    DB_NAME = 'news_app'
    DB_TABLE_SHORT = 'news'
    DB_TABLE_WHOLE = 'news_whole_content'
    
class NewsCategoryClassifier(nn.Module):
    def __init__(self, bertModel, out_feat, freeze_bert):
        super().__init__()
        D_in, H, D_out = bertModel.config.hidden_size, 100, out_feat
        self.bert = bertModel
        self.classifier = nn.Sequential(
            nn.Linear(D_in,H),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(H, D_out)
        )
        # Freeze the pre-trained layers of BERT model
        if freeze_bert==True:
          for param in self.bert.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask )
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)
        return logits

demo_text_news = "An annual celebration took on a different feel as Russia's invasion dragged into Day 206."
label = "Politics"

def get_all_text(df, col):
    text = ' '.join(str(el) for el in df[col] if el != ' ')
    text = re.sub(' +', ' ', text)
    return text


@st.cache(show_spinner=True)
def show_wordcloud(df, col):
    wc = WordCloud(collocations=False, stopwords=STOPWORDS,
               background_color="white", max_words=1000,
               max_font_size=256, random_state=42,
               width=1400, height=400)

    text=get_all_text(df, col)
    wc.generate(text)
    return wc
    
def load_model(name):
    if name=='classifier':
        model = torch.load('classifier_v1.pt',map_location='cpu')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        model.eval()
    elif name=='pegasus_sum':
        model = TFPegasusForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_sum_model.pt")
        tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_sum_tokenizer.pt")
    elif name=='pegasus_x':
        model = TFPegasusForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_x_model.pt")
        tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_x_tokenizer.pt")
    elif name=='flan':
        model = T5ForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/flan_model.pt")
        tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/flan_tokenizer.pt")
        model.eval()
    else:
        print('Wrong name provided')
        return None
    return model, tokenizer


def preprocess(text):
    new_text = []
    if type(text)==float:
      print(text)
    for t in text.split(" "):
        t = 'HTTPURL' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)


def id2label_classifier(id):
    return 'U.S. NEWS'


def tokenize_function(text, tokenizer):
    tok = tokenizer(text,add_special_tokens=True, padding="max_length", 
                    max_length = Config.MAX_TEXT_LEN_CAT,truncation=True, 
                    return_tensors="pt")
    return tok['input_ids'], tok['attention_mask']

def infer_from_model(text, model, tokenizer, model_type):
    cleaned_text = preprocess(text)
    inputs_ids, attention_mask = tokenize_function(cleaned_text, tokenizer=tokenizer)
    if model_type == 'classifier':
        logits = model(inputs_ids, attention_mask)
        prediction = torch.argmax(torch.abs(logits), dim=1)
        return id2label_classifier(prediction.item())
    elif  model_type=='headline_pegasus':
        tokens = tokenizer(text, return_tensors='tf', max_length=Config.MAX_LEN_HEADLINE, padding='max_length', truncation=True)
        model_prediction = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        decodes = tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decodes
    elif  model_type=='headline_flan':
        tokens = tokenizer(text, return_tensors='pt', max_length=Config.MAX_LEN_HEADLINE, padding='max_length', truncation=True)
        model_prediction = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        decodes = tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decodes
    else: #summary pegasus
        tokens = tokenizer(text, return_tensors='tf', max_length=Config.MAX_LEN_HEADLINE, padding='max_length', truncation=True)
        model_prediction = model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        decodes = tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return decodes

def getCollectionFromDB(db_name, collection_name):
    db_client = MongoClient('localhost', 27017)
    db = db_client[db_name]
    collection = db[collection_name]
    return collection

def selectFromDB(collection, table, option):
    '''
    :param table: numele tabelei
    :param collection: numele bazei de date 
    :param option: numarul de inregistrari de luat din bd
    :return df: dataframe cu ultimele option inregistrari, daca option=-1 se vor intoarce toate inregistrarile
    '''
    collection = getCollectionFromDB(collection, table)
    documents = []
    if option == -1:
        result = collection.find()
    else:
        result = collection.find().sort('$natural', pymongo.DESCENDING).limit(option)
    for doc in result:
        documents.append(doc)
    if documents:
        df = pd.DataFrame(documents)
        return df
    else:
        return ""


# df_short = selectFromDB(Config.DB_NAME,Config.DB_TABLE_SHORT, -1)
# print(df_short)
# df_short = df_short.drop(columns=['_id'])

df_whole = selectFromDB(Config.DB_NAME,Config.DB_TABLE_WHOLE, -1)
print(df_whole)
df_whole = df_whole.drop(columns=['_id'])


st.set_page_config(page_icon="🐤", page_title="News App Processor Engine")
st.sidebar.image("logo.jpg", use_column_width=True)

model_classifier, tokenizer_classifier= load_model('classifier')
model_flan_headline, tokenizer_flan_headline= load_model('flan')
model_pegasus_headline, tokenizer_pegasus_headline = load_model('pegasus_x')
model_pegasus_sum, tokenizer_pegasus_sum = load_model('pegasus_sum')
    
st.sidebar.subheader('Analyize News')
text_introduced = st.sidebar.text_input('Type', placeholder=demo_text_news)
check_class = st.sidebar.button(label='Classify')
check_headline_pegasus = st.sidebar.button(label='Generate Headline with Pegasus')
check_headline_flan = st.sidebar.button(label='Generate Headline with Flan')
check_summary = st.sidebar.button(label='Generate Summary')


if check_class:
    result =  infer_from_model(text_introduced, model_classifier, tokenizer_classifier,'classifier')
    st.sidebar.write(result)
if check_headline_flan:
    result =  infer_from_model(text_introduced, model_flan_headline, tokenizer_flan_headline,'headline_flan')
    st.sidebar.write(result)
if check_headline_pegasus:
    result = infer_from_model(text_introduced,model_pegasus_headline, tokenizer_pegasus_headline,'headline_pegasus')
    st.sidebar.write(result)  
if check_summary:
    result = infer_from_model(text_introduced, model_pegasus_sum, tokenizer_pegasus_sum,'summary')
    st.sidebar.write(result)
st.write('<base target="_blank">', unsafe_allow_html=True)
prev_time = [time.time()]

a, b = st.columns([1, 10])

with a:
    st.text("")
    st.image("logo.jpg", width=50)
with b:
    st.title("News App Processor Eninge")

st.header('News Hub')
st.dataframe(df_whole)

treshold = int(st.number_input('Insert the maximum number of news to analyze: ',min_value=1,max_value=500,step=5))
if treshold > 0:
    df_limited = df_whole[:treshold]
    df_limited.drop(columns=['authors','date'], inplace=True)
    cols_shows = st.columns([1,1,1,1])
    show_category_for_batch = cols_shows[0].button(label="Get category")
    show_summary_for_batch = cols_shows[1].button(label="Generate summary")
    show_headline_for_batch = cols_shows[2].button(label='Generate headline')
    show_all_for_batch = cols_shows[3].button(label='Get all')
    
    if show_category_for_batch:
        df_limited['predicted_category'] = df_limited.apply(lambda x: infer_from_model(x['short_description'],model_classifier,tokenizer_classifier, 'classifier'),axis=1)
        df_limited.drop(columns=['category','short_description'],inplace=True)
        st.dataframe(df_limited)
    if show_summary_for_batch:
        df_limited = df_whole
        df_limited['generated_summary'] = df_limited.apply(lambda x: infer_from_model(x['whole_article'],model_pegasus_sum,tokenizer_pegasus_sum, 'summary'),axis=1)
        # df_limited.drop(columns=['category','short_description'],inplace=True)
        st.dataframe(df_limited)
    if show_headline_for_batch:
        df_limited = df_whole
        df_limited['generated_headline'] = df_limited.apply(lambda x: infer_from_model(x['whole_article'],model_pegasus_headline,tokenizer_pegasus_headline, 'headline_pegasus'),axis=1)
        df_limited.drop(columns=['category','short_description'],inplace=True)
        st.dataframe(df_limited)
    if show_all_for_batch:
        df_limited['predicted_category'] = df_limited.apply(lambda x: infer_from_model(x['whole_article'],model_classifier,tokenizer_classifier, 'classifier'),axis=1)
        df_limited['generated_summary'] = df_limited.apply(lambda x: infer_from_model(x['whole_article'],model_pegasus_sum,tokenizer_pegasus_sum, 'summary'),axis=1)
        df_limited['generated_headline'] = df_limited.apply(lambda x: infer_from_model(x['whole_article'],model_pegasus_headline,tokenizer_pegasus_headline, 'headline_pegasus'),axis=1)
        # df_limited.drop(columns=['category','short_description'],inplace=True)
        st.dataframe(df_limited)


show_wc = st.checkbox('Show Wordcloud from all headlines')
if show_wc:
    wc = show_wordcloud(df_whole, 'headline')
    fig = plt.figure(figsize=(20, 10), facecolor='k')
    plt.title(f'Wordcloud')
    plt.imshow(wc, interpolation='bilInear')
    plt.axis("off")
    plt.tight_layout(pad=0)   
    st.pyplot(fig)
