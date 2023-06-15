from transformers import AutoTokenizer, TFPegasusForConditionalGeneration, T5ForConditionalGeneration
from tqdm import tqdm
import json
import torch
DB_COLLECTION_NAME ='news_whole_content'

pegasus_x_model = TFPegasusForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_x_model.pt")
pegasus_x_tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_x_tokenizer.pt")

pegasus_sum_model = TFPegasusForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_sum_model.pt")
pegasus_sum_tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/pegasus_sum_tokenizer.pt")

flan_t5_model = T5ForConditionalGeneration.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/flan_model.pt")
flan_t5_tokenizer = AutoTokenizer.from_pretrained("E:/master/anul1/sem22/pdi/proiect news app/flan_tokenizer.pt")


def generate_headline_pegasus(article_list):
    list_string = article_list
    decodes_list = []
    batch_size=16
    for i in tqdm(range(0, len(list_string), batch_size)):

        end_interval = min(i+batch_size, len(list_string))

        tokens = pegasus_x_tokenizer(list_string[i:end_interval], return_tensors='tf', max_length=1024, padding='max_length', truncation=True)
        model_prediction = pegasus_x_model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        decodes = pegasus_x_tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decodes_list += decodes
    return decodes_list

def generate_summary_pegasus(article_list):
    list_string = article_list
    decodes_list = []
    batch_size=16
    for i in tqdm(range(0, len(list_string), batch_size)):

        end_interval = min(i+batch_size, len(list_string))

        tokens = pegasus_sum_tokenizer(list_string[i:end_interval], return_tensors='tf', max_length=1024, padding='max_length', truncation=True)
        model_prediction = pegasus_sum_model.generate(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
        decodes = pegasus_sum_tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decodes_list += decodes
    return decodes_list

def generate_headline_t5(article_list):
    list_string = [f"Headline: {art}" for art in article_list]
    decodes_list = []
    batch_size=16
    for i in tqdm(range(0, len(list_string), batch_size)):

        end_interval = min(i+batch_size, len(list_string))

        tokens = flan_t5_tokenizer(list_string[i:end_interval], return_tensors='pt', max_length=1024, padding='max_length', truncation=True)
        model_prediction = flan_t5_model.generate(input_ids=tokens['input_ids'].to(torch_device), attention_mask=tokens['attention_mask'].to(torch_device))
        decodes = flan_t5_tokenizer.batch_decode(model_prediction, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        decodes_list += decodes
    return decodes_list



# exit()
# news = json.load(open('news_data_1000.json', 'r'))
# article_list = [n['whole_article'] for n in news]
# ref_headlines = [n['headline'] for n in news]

# t5_headlines = generate_headline_t5(article_list)
# pegasus_headlines = generate_headline_pegasus(article_list)

# result = {'ref_headlines': ref_headlines, 't5_headlines': t5_headlines, 'pegasus_headlines': pegasus_headlines}

# json.dump(result, open('result_news_data_1000.json', 'w'))