"""

Поиск чанков и подача их в контекст, вместо подачи ответа
чтобы сохранялся контекст ответа. И модель видела этот контекст

"""
from difflib import SequenceMatcher
from sentence_transformers.util import cos_sim
from sentence_transformers import CrossEncoder
from telebot import types
from time import sleep


import json
import pandas as pd
import pickle
import psycopg2
import random
import requests
import telebot
import torch

private_vars = {
    "TG_POMOSHNIK_BOT_TOKEN": '6836506185:AAEAvwcZEniKjOdvgsTnqPlJ-qQPMn9i2Mc',
    "URL_VECTORISATION": "http://ext-delivery-bert-cl1.dl.wb.ru:8081/vector",
    "URL_GET_ANSWER": "https://bert.wb.ru/api/get-answer",
    "URL_LLM": 'http://ext-delivery-search-llm-02.el.wb.ru:8082/generate_answer_without_prompt',
    "URL_SCORE_ANSWER": "https://bert.wb.ru/api/score-answer",
    "GET_ANSWER_X_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzaWQiOiIzOTA2NjM6Mjc3ZTFmNDNmM2UyNDhlMCIsInVpZCI6MzkwNjYzLCJ3dWlkIjo2MzI5MTMzOSwicGlkIjowLCJ4cGlkIjowLCJzIjowLCJyIjpbNF0sImFkbWluIjp0cnVlLCJhZG1pbl9yb2xlcyI6WyJhZG1pbiIsImludGVybmFsIl0sImlzX3JlZnJlc2giOmZhbHNlLCJyZWZyZXNoX2tleSI6IiIsImNvdW50cnkiOiIiLCJhcGlkcyI6bnVsbCwiZWFwaWRzIjpudWxsLCJ1c2hzIjpudWxsLCJ2ZXJzaW9uIjoxLCJkZXZpY2VfdXVpZCI6IjAwMThjMzljMzg4ZWVlMTA1YmVkOTA3ZDkzYzg1ZWFiIiwiZXhwIjoxNjk2NzcwOTgyLCJpYXQiOjE2OTU5MDY5ODJ9.Q6sCYNy1TbUwlzm0b7g0zCHVa3ug8HwLlLPJFF5eP5mbphGuY9l165XpJ0LFMSrw_uFPOpzJMlDrddb1Yi-X1hbO2oauu0sOWS7qsEsuhaweWPxlVFrzlJNdiF1CETiSb8zdjYXavS-ELYz_B_VaTtLiTVmtlo_bAgMwm7uNROOYMX6kxtt1COGV9xTLYYiUAlvbDNW-IDf3jKFypMMxBEDyJRUYUPGXMNGOt2jVhaXFZXDP0sHKXf6mj0rEX5ACRgGRMAwYf_WfZC_ZXGCcSUBEGtVNolKIPUvrFj0JszTJhfIDj-YL0O3RWmQ90TifMEoVkxO0mW8LrNXaEBCTgA",
    "GET_SCORE_X_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzaWQiOiIzOTA2NjM6Mjc3ZTFmNDNmM2UyNDhlMCIsInVpZCI6MzkwNjYzLCJ3dWlkIjo2MzI5MTMzOSwicGlkIjowLCJ4cGlkIjowLCJzIjowLCJyIjpbNF0sImFkbWluIjp0cnVlLCJhZG1pbl9yb2xlcyI6WyJhZG1pbiIsImludGVybmFsIl0sImlzX3JlZnJlc2giOmZhbHNlLCJyZWZyZXNoX2tleSI6IiIsImNvdW50cnkiOiIiLCJhcGlkcyI6bnVsbCwiZWFwaWRzIjpudWxsLCJ1c2hzIjpudWxsLCJ2ZXJzaW9uIjoxLCJkZXZpY2VfdXVpZCI6IjAwMThjMzljMzg4ZWVlMTA1YmVkOTA3ZDkzYzg1ZWFiIiwiZXhwIjoxNjk2NzcwOTgyLCJpYXQiOjE2OTU5MDY5ODJ9.Q6sCYNy1TbUwlzm0b7g0zCHVa3ug8HwLlLPJFF5eP5mbphGuY9l165XpJ0LFMSrw_uFPOpzJMlDrddb1Yi-X1hbO2oauu0sOWS7qsEsuhaweWPxlVFrzlJNdiF1CETiSb8zdjYXavS-ELYz_B_VaTtLiTVmtlo_bAgMwm7uNROOYMX6kxtt1COGV9xTLYYiUAlvbDNW-IDf3jKFypMMxBEDyJRUYUPGXMNGOt2jVhaXFZXDP0sHKXf6mj0rEX5ACRgGRMAwYf_WfZC_ZXGCcSUBEGtVNolKIPUvrFj0JszTJhfIDj-YL0O3RWmQ90TifMEoVkxO0mW8LrNXaEBCTgA",
    "DB_USER": "bert",
    "DB_PASSWORD": "b6d8n1MD6ULsjmWo4PhToB2aU5QsKDCz",
    "DB_HOST": "ext-delivery-bert-pgsql-cl1-haproxy.ext-delivery.svc.k8s.prod-dl",
    "DB_DATABASE_NAME": "bert",
    "DB_PORT_WRITE": "5000"
}

# Model init
DEFAULT_MESSAGE_TEMPLATE = "<s>{role}\n{content}</s>"
DEFAULT_RESPONSE_TEMPLATE = "<s>bot\n"
DEFAULT_SYSTEM_PROMPT = "Ты — PointGPT, русскоязычный автоматический ассистент. Ты разговариваешь с людьми и помогаешь им."

# Private data
TOKEN = private_vars['TG_POMOSHNIK_BOT_TOKEN']
URL_VECTORISATION = private_vars['URL_VECTORISATION']
URL_SCORE_ANSWER = private_vars['URL_SCORE_ANSWER']
URL_LLM = private_vars['URL_LLM']


# headers for URL_SCORE_ANSWER
headers_get_score = {
    'x-token': private_vars['GET_SCORE_X_TOKEN'],
    'Content-Type': 'application/json'
}


# Bot texts
START_TEXT = "Привет, я Помощник, отвечаю на вопросы по работе в программе Point!\nЧтобы начать просто задай вопрос!"
GET_SCORE_TEXT = "Пожалуйста, оцените ответ Помощника, это поможет стать ему лучше!"
LIKE_TEXT = "Спасибо за оценку!"
DISLIKE_TEXT = "Спасибо за оценку!\nЕсли вы не получили ответ на свой вопрос, то попробуйте задать его иначе."
PNZO_ANSWER = "Пока не знаю ответ на этот вопрос.\nЗадавай больше вопросов, чтобы я стал умнее 😁"
CORRECT_ANSWER = "Пожалуйста, сформулируйте корректный вопрос."


# DBeaver connection
class Connection:
    """
    Class for connection to DataBase
    """

    def __init__(self):
        # Initializaion
        print('Start Conncetion')
        self.conn = psycopg2.connect(
            host=private_vars['DB_HOST'],
            port=private_vars['DB_PORT_WRITE'],
            database=private_vars['DB_DATABASE_NAME'],
            user=private_vars['DB_USER'],
            password=private_vars['DB_PASSWORD'],
            sslmode="disable"
        )
        self.curr = self.conn.cursor()
        print('Connection succeed')
    def __del__(self):
        # Close connection
        self.curr.close()
        self.conn.close()
        print('Connection closed')


# Class for model init
class Conversation:
    def __init__(
        self,
        message_template=DEFAULT_MESSAGE_TEMPLATE,
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        response_template=DEFAULT_RESPONSE_TEMPLATE,
    ):
        self.message_template = message_template
        self.response_template = response_template
        self.messages = [{
            'role': 'system',
            'content': system_prompt
        }]

    def add_user_message(self, message):
        self.messages = self.messages[-4:]
        self.messages.append({
            "role": "user",
            "content": message
        })

    def add_bot_message(self, message):
        self.messages.append({
            "role": "bot",
            "content": message
        })

    def pnzo_in_line(self):
        bot_messages = [msg for msg in self.messages if msg['role'] == 'bot']
        print(bot_messages)

    def get_prompt(self):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


def add_question_answer_to_database(user_id: int, question_id: int, answer_id: int, question: str, answer: str):
    """
    Add question and answer to dataset telegram_bot_messages
    - Parameters:
        1. user_id (int): user telegram id
        2. question_id (int): question message id
        3. answer_id (int): answer message id
        4. question (str): question text
        5. answer (str): answer text
    - Return:
    """
    c = Connection()
    conn, curr = c.conn, c.curr
    row = tuple([user_id, question_id, answer_id, question, answer])
    curr.execute('insert into telegram_bot_messages (user_id, question_msg_id, answer_msg_id, question, answer) \
                    values (%s, %s, %s, %s, %s);', row)

    conn.commit()


def check_answer_for_pnzo(answer: str) -> bool:
    """
    Check answer for similarity with pnzo or welcome text to don't send user a score buttons
    - Parameters:
        1. answer (str): answer text
    - Return:
        2. flag (bool): flag if answer like texts that need a score buttons
    """
    return SequenceMatcher(None, answer, PNZO_ANSWER).ratio() > 0.9 or \
        SequenceMatcher(None, answer, CORRECT_ANSWER).ratio() > 0.9 or \
        answer.lower().startswith('привет')


def get_bert_vector(text: str) -> list:
    """
    :param text:
    :return:
    """
    req_json = {'search_string': text}
    response_json = requests.post(URL_VECTORISATION, json=req_json)
    bert_vector = response_json.json()['vector']

    return bert_vector

"""
def get_chunks(question):
    c = Connection()
    conn, curr = c.conn, c.curr
    curr.execute(f'select * from finder_documents;')
    cursor_columns = [col.name for col in curr.description]
    data = pd.DataFrame(curr.fetchall(), columns=cursor_columns)

    question_vect = eval(get_bert_vector(question * 2))
    data['cos_sim'] = data['vector'].apply(
        lambda x: cos_sim(question_vect, eval(x)).item())
    return list(data.sort_values(by='cos_sim', ascending=False).head(5)['message'])
"""

def get_chunks(question):
    c = Connection()
    conn, curr = c.conn, c.curr
    curr.execute(f'select * from finder_documents;')
    cursor_columns = [col.name for col in curr.description]
    data = pd.DataFrame(curr.fetchall(), columns=cursor_columns)

    question_vect = eval(get_bert_vector(question * 2))
    data['cos_sim'] = data['vector'].apply(
        lambda x: cos_sim(question_vect, eval(x)).item())
    print(data.head(5))
    print("zaebis_chanks')
    data = data.sort_values(by='cos_sim', ascending=False).head(50)
    data['score'] = data['message'].apply(lambda x: model_search.predict([(question, x)]))
    
    chunks= list(data.sort_values(by='score', ascending=False).head(5)['message'])
    print(chunks)
    return chunks


def get_question_answer_from_database(user_id: int, question_id: int, answer_id: int) -> tuple:
    """
    Get answer and question texts from telegram_bot_messages dataset
    - Parameters:
        1. user_id (int): telegram user id
        2. question_id (int): id of question message id
        3. answer_id (int): id of answer message id
    - Return:
        1. question, answer (str, str): tuple of question and answer
            for user_id, question_id and answer_id from telegram_bot_messages dataset
    """
    c = Connection()
    conn, curr = c.conn, c.curr
    curr.execute(f'select * from telegram_bot_messages where user_id = {user_id} \
                    and question_msg_id = {question_id} and answer_msg_id = {answer_id}')
    data = curr.fetchall()
    return data[0][3], data[0][4]


def get_score_request(question: str, answer: str, relevant: bool):
    """
    Get request for the score service
    - Parameters:
        1. question (str): question text
        2. answer (str): answer text
        3. relevant (bool): True if it was like
    - Return:
    """
    payload = {
        'request': question,
        'response': answer,
        'relevant': relevant
    }
    response = requests.post(
        URL_SCORE_ANSWER, headers=headers_get_score, data=json.dumps(payload))
    if response.status_code != 200:
        print('Response:', response)


def script_pomoshnik(user_id: int, question: str, conversation, message_id: int, chat_id: int) -> str:
    chunks = '\n'.join(get_chunks(question))
    # 1st generation
    prompt = f"'{chunks}'\nПо контексту выше, ответь кратко: '{question}'"
    answer = generate(prompt)
    bot.edit_message_text(
        chat_id=chat_id, text='Спрашиваю у отделов...', message_id=message_id)

    # 2nd generation
    prompt = f"'{answer}\n{chunks}'\nПо контексту выше, ответь кратко: '{question}'"
    answer = generate(prompt)
    bot.edit_message_text(
        chat_id=chat_id, text='Уточняю у директора...', message_id=message_id)

    if check_answer_for_pnzo(answer) or len(answer) == 0:
        conversation.add_user_message(question)
        prompt = conversation.get_prompt()
        output = generate(prompt)
        text = PNZO_ANSWER + '\n\n' + \
            'Возможно вам подойдет ответ: ' + output.split('bot')[0]
        bot.edit_message_text(chat_id=chat_id, text=text,
                              message_id=message_id)
    else:
        conversation.add_user_message(prompt)
        output = answer
        bot.edit_message_text(
            chat_id=chat_id, text=answer, message_id=message_id)
    return output


def script_chat(user_id, question, conversation, message_id, chat_id) -> str:
    bot.edit_message_text(
        chat_id=chat_id, text='Спрашиваю у отделов...', message_id=message_id)
    conversation.add_user_message(question)
    prompt = conversation.get_prompt()
    output = generate(prompt)
    bot.edit_message_text(chat_id=chat_id, text=output, message_id=message_id)

    return output


# script classifier
model_clf = pickle.load(open("model_script_clf.pickle", "rb"))
model_search = CrossEncoder('cross-encoder/stsb-roberta-large')

bot = telebot.TeleBot(TOKEN)

# global var for Conversations class objects
conversations = {}


def generate(question: str) -> str:
    """
    Make request for generate_answer service
    - Parameters:
        1. question (srt): question text
    - Return:
        1. response (str): response from the service
    """
    payload = {
        'context': question,
        'question': ''
    }
    response = requests.post(
        URL_LLM, json=payload)
    if response.status_code != 200:
        return CORRECT_ANSWER
    return eval(response.text)['answer']


def remake_answer(answer):
    return ' '.join(answer.split('?')[1:])


# Обработчик команды /start
@bot.message_handler(commands=['start'])
def start(message):
    user_id = message.from_user.id
    bot.send_message(user_id, START_TEXT)
    conversations[user_id] = Conversation()


# Обработчик для каждого нового сообщения
@bot.message_handler(content_types=['text'])
def handle_question(message):
    user_id = message.from_user.id
    question = message.text
    msg_log = bot.send_message(user_id, 'Обрабатываю запрос...')

    # conversation init
    if user_id not in conversations:
        conversations[user_id] = Conversation()
    conversation = conversations[user_id]

    # scipt classification
    if model_clf.predict([question]):
        output = script_chat(user_id, question, conversation,
                             msg_log.message_id, message.chat.id)
    else:
        output = script_pomoshnik(
            user_id, question, conversation, msg_log.message_id, message.chat.id)

        # add buttons for scoring
        if not check_answer_for_pnzo(output):
            keyboard = types.InlineKeyboardMarkup(row_width=2)
            callback_button1 = types.InlineKeyboardButton(f'👍', callback_data=f"like_{user_id}_{message.message_id}_{msg_log.message_id}")
            callback_button2 = types.InlineKeyboardButton(f'👎', callback_data=f"dislike_{user_id}_{message.message_id}_{msg_log.message_id}")
            keyboard.add(callback_button1, callback_button2)
            bot.send_message(user_id, GET_SCORE_TEXT, reply_markup=keyboard)

    add_question_answer_to_database(
        user_id, message.message_id, msg_log.message_id, question, output)

    conversation.add_bot_message(output)


# Обработчик нетекстовых запросов
@bot.message_handler(content_types=['photo', 'document', 'audio', 'voice', 'video', 'sticker'])
def handle_message(message):
    """
    Handler for not text messages 
    """
    stickers = [
        'CAACAgIAAxkBAAJnbGV56uJjmVcVEbrMIxY2aH2ouMPiAAJMBQACIwUNAAFlusszZpV-2jME',
        'CAACAgIAAxkBAAJnVmV56OJn59d_Bn1kg6uuLwFabN6aAAJSEwACMW45SAS_cwdjHnOTMwQ',
        'CAACAgIAAxkBAAJnXmV56nt4kakD-mjdfBZU5evvRFGTAAJwBQACIwUNAAEK2cEGvypKtTME',
        'CAACAgIAAxkBAAJnXGV56nmeE8cw7ZTunhDV7AdpbKY7AAJVBQACIwUNAAH6DXJLwg6K8jME',
        'CAACAgQAAxkBAAJnhGV563gofyi4rBtiKyo1o4mOcAXdAAKoAAPOOQgN2hWbG1Xxf5YzBA'
    ]
    num = random.randint(0, len(stickers)-1)
    user_id = message.from_user.id
    bot.send_sticker(
        user_id, stickers[num])
    bot.send_message(
        user_id, 'К сожалению, я пока понимаю только текст. Попробуйте переписать свой вопрос.')


@bot.callback_query_handler(func=lambda call: True)
def handle_button_click(call):
    """
    Handler for button clicks (like/dislike)
    """
    user_id = call.data.split('_')[1]
    question_id = call.data.split('_')[2]
    answer_id = call.data.split('_')[3]

    try:
        question, answer = get_question_answer_from_database(
            user_id, question_id, answer_id)
    except:
        print(f'Problems with {user_id}, {question_id}, {answer_id} -- get_question_answer_from_database()')

    if call.data.startswith('like_'):
        # like handler
        bot.edit_message_text(chat_id=call.message.chat.id,
                              text=LIKE_TEXT, message_id=call.message.message_id)
        try:
            get_score_request(question=question, answer=answer, relevant=True)
        except:
            print(f'Problems with {user_id}, {question_id}, {answer_id} -- get_score_request()')

    elif call.data.startswith('dislike_'):
        # dislike handler
        bot.edit_message_text(chat_id=call.message.chat.id,
                              text=DISLIKE_TEXT, message_id=call.message.message_id)
        try:
            get_score_request(question=question, answer=answer, relevant=False)
        except:
            print(f'Problems with {user_id}, {question_id}, {answer_id} -- get_score_request()')


# bot.polling()

# Start bot
while True:
    try:
        bot.polling(none_stop=True)
    except Exception as e:
        conversations = {}
        print(e)
        sleep(5)