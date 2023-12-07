from transformers import CanineTokenizer, CanineForTokenClassification, BertTokenizer, BertForTokenClassification, AutoTokenizer, LukeForTokenClassification, pipeline
import pandas as pd
import time
import re

def load_model_CANINE(model_pth, device, label2id, id2label):
    tokenizer = CanineTokenizer.from_pretrained(model_pth)
    model = CanineForTokenClassification.from_pretrained(model_pth, 
                                            num_labels=len(id2label),
                                            id2label=id2label,
                                            label2id=label2id)
    return tokenizer, model.to(device)

def load_model_BERT(model_pth, device, label2id, id2label):
    model = BertForTokenClassification.from_pretrained(model_pth, 
                                        num_labels=len(id2label),
                                        id2label=id2label,
                                        label2id=label2id)
    tokenizer = BertTokenizer.from_pretrained(model_pth)
    return tokenizer, model.to(device)

def load_model_LUKE(model_pth, device, label2id, id2label):
    model = LukeForTokenClassification.from_pretrained(model_pth, 
                                        num_labels=len(id2label),
                                        id2label=id2label,
                                        label2id=label2id)
    tokenizer = AutoTokenizer.from_pretrained(model_pth)
    return tokenizer, model.to(device)

def inference(offer, model, tokenizer, device):
    pipe = pipeline(task="ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple", device=device)
    return pipe(offer)

def brand_line_width_height_radius(result):
    # Создаем словари для каждой сущности
    entities = {}
    for item in result:
        entity_group = item['entity_group']
        word = item['word']
        score = item['score']
        if score > 0.7:
            if entity_group in entities:
                entities[entity_group].append(word)
            else:
                entities[entity_group] = [word]

    for key, value in entities.items():
        entities[key] = ''.join(value)
    return entities

def apply_on_df(model, tokenizer, df, device, column = 'offer'):
    for index, row in df.iterrows():
        offer = row[column]
        result = inference(offer, model, tokenizer, device)
        entities = brand_line_width_height_radius(result)

        # Заполнение DataFrame
        df.at[index, 'brand_pred'] = entities.get('brand', '')
        df.at[index, 'width_pred'] = entities.get('width', '')
        df.at[index, 'height_pred'] = entities.get('height', '')
        df.at[index, 'radius_pred'] = entities.get('radius', '')
        df.at[index, 'line_pred'] = entities.get('line', '')
        df.at[index, 'v_ind_pred'] = entities.get('v_ind', '')
    return df

def process_text(input_text):
    def separate_letters_and_numbers(input_text):
        # Используем регулярное выражение для поиска сочетаний букв и цифр
        pattern = re.compile(r'(\D+|\d+)')
        
        # Используем findall для нахождения всех сочетаний
        matches = pattern.findall(input_text)
        # Возвращаем строку с пробелами между буквами и цифрами
        return ' '.join(matches)

    processed_text = separate_letters_and_numbers(input_text)    
    processed_text = processed_text.replace("|", " | ")
    processed_text = processed_text.replace("(", " ( ")
    processed_text = processed_text.replace(")", " ) ")
    processed_text = processed_text.replace("[", " [ ")
    processed_text = processed_text.replace("]", " ] ")
    # Убираем повторяющиеся пробелы
    processed_text = re.sub(r'  +', ' ', processed_text)
    return processed_text

def process_digits(txt):
        return re.sub(r"[^0123456789A-Za-z]","", txt)

def process_brand_line(txt):
    return re.sub(r"[^0123456789A-Za-zА-Яа-я/ ]","", txt)

if __name__ == "__main__":

    src = [
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/Gislaved.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/Nordman.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/Pirelli.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/Yokohama.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/Кама.xlsx"
        ]

    dst = [
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/luke-base-241123_2/Gislaved_Igor241123_2.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/luke-base-241123_2/Nordman_Igor241123_2.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/luke-base-241123_2/Pirelli_Igor241123_2.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/luke-base-241123_2/Yokohama_Igor241123_2.xlsx",
    "/home/sondors/Documents/price/BERT_NER/csv_to_label/luke-base-241123_2/Кама_Igor241123_2.xlsx"
        ]

    column_name = ['PRICE_NAME', 'PRICE_NAME', 'Unnamed: 6', 'PRICE_NAME', 'PRICE_NAME']

    label2id = {'B-width': 1,
                'B-height': 2, 
                'B-radius': 3, 
                'I-radius': 4,
                'B-brand': 5, 
                'B-line': 6, 
                'I-line': 7,
                'B-v_ind': 8,
                'I-v_ind': 9,
                'O': 0}
        
    id2label = {1: 'B-width',
                2: 'B-height', 
                3: 'B-radius', 
                4: 'I-radius',
                5: 'B-brand', 
                6: 'B-line', 
                7: 'I-line',
                8: 'B-v_ind',
                9: 'I-v_ind',
                0: 'O'}

    device = "cuda:0"

    # model_pth = "/home/sondors/CANINE-epoch_4"
    # tokenizer_CANINE, model_CANINE = load_model_CANINE(model_pth, device, label2id, id2label)

    model_pth = "/home/sondors/luke-base-epoch_5"
    tokenizer, model = load_model_LUKE(model_pth, device, label2id, id2label)

    for i in range(len(src)):
        start_time = time.time()

        df_original = pd.read_excel(src[i], dtype=str)
        df = pd.DataFrame()
        df['PRICE_NAME'] = df_original[column_name[i]]

        df_BERT = df.copy()
        df_BERT['PRICE_NAME'] = df_BERT['PRICE_NAME'].apply(process_text)

        df_BERT = apply_on_df(model, tokenizer, df_BERT, device, column = 'PRICE_NAME')

        df_original['width_pred'] = df_BERT['width_pred'].apply(process_digits)
        df_original['height_pred'] = df_BERT['height_pred'].apply(process_digits)
        df_original['radius_pred'] = df_BERT['radius_pred'].apply(process_digits)
        df_original['v_ind_pred'] = df_BERT['v_ind_pred'].apply(process_digits)

        # df_original['brand_pred'] = df_BERT['brand_pred'].apply(process_brand_line)
        df_original['line_pred'] = df_BERT['line_pred'].apply(process_brand_line)

        def del_short_str(df, columns):
            for column in columns:
                df[column] = df[column].apply(lambda x: '' if (isinstance(x, str) and len(str(re.sub(r"[^0123456789]","", x))) < 2) else x)
            return df
    
        columns_to_check = ['width_pred', 'height_pred', 'radius_pred', 'v_ind_pred']
        df_original = del_short_str(df_original, columns_to_check)

        df_original.to_excel(dst[i])

        print(df_original)
        print(f"time_spent = {time.time() - start_time}")