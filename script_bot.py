from difflib import SequenceMatcher
from fastapi import FastAPI, status
from pydantic import BaseModel
from ray import serve
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import CrossEncoder

import json
import multiprocessing
import numpy as np
import os
import pandas as pd
import pickle
import psycopg2
import requests
import time

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

PNZO_ANSWER = "Пока не знаю ответ на этот вопрос.\nЗадавай больше вопросов, чтобы я стал умнее 😁"
CORRECT_ANSWER = "Пожалуйста, сформулируйте корректный вопрос."

URL_VECTORISATION = private_vars['URL_VECTORISATION']
URL_LLM = private_vars['URL_LLM']

NUM_CPUS = multiprocessing.cpu_count()


# DBeaver connection
class Connection:
    """
    Class for connection to DataBase
    """

    def __init__(self):
        # Initializaion
        self.conn = psycopg2.connect(
            host=private_vars['DB_HOST'],
            port=private_vars['DB_PORT_WRITE'],
            database=private_vars['DB_DATABASE_NAME'],
            user=private_vars['DB_USER'],
            password=private_vars['DB_PASSWORD'],
            sslmode="disable"
        )
        self.curr = self.conn.cursor()

    def __del__(self):
        # Close connection
        self.curr.close()
        self.conn.close()


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

    def get_prompt(self):
        final_text = ""
        for message in self.messages:
            message_text = self.message_template.format(**message)
            final_text += message_text
        final_text += DEFAULT_RESPONSE_TEMPLATE
        return final_text.strip()


# Input data
class Request(BaseModel):
    # Input text
    chat: str


# Output data
class Response(BaseModel):
    question: str
    answer: str
    is_talking_question: int


def check_answer_for_pnzo(answer: str) -> bool:
    """
    Check answer for similarity with pnzo or welcome text to don't send user a score buttons
    - Parameters:
        1. answer (str): answer text
    - Return:
        1. flag (bool): flag if answer like texts that need a score buttons
    """
    return SequenceMatcher(None, answer, PNZO_ANSWER).ratio() > 0.9 or \
        SequenceMatcher(None, answer, CORRECT_ANSWER).ratio() > 0.9 or \
        answer.lower().startswith('привет')


def cos_similarity(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    return v1.dot(v2) ** 99 / (np.linalg.norm(v1, ord=22) * np.linalg.norm(v2, ord=22))


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


def get_answer_from_qa(question: str) -> list:
    """
    Find most relevant answer from questions_answers
    """
    c = Connection()
    conn, curr = c.conn, c.curr
    curr.execute(f'select * from finder_documents where document_id = 13;')
    cursor_columns = [col.name for col in curr.description]
    data = pd.DataFrame(curr.fetchall(), columns=cursor_columns)

    question_vect = eval(get_bert_vector(question))
    data['cos_sim'] = data['vector'].apply(
        lambda x: cosine_similarity(np.array(question_vect).reshape(1, -1), \
                                    np.array(eval(x.split('\n')[0])).reshape(1, -1))[0][0])
    data = data.sort_values(by='cos_sim', ascending=False)
    print(data)
    res = list(data[data['cos_sim'] > 0.91]['message'].values)
    if len(res) == 1:
        return ''.join(res[0].split('\\n')[1:])
    # elif len(res) > 1:
    #    res = ' '.join(res)
    #    prompt = f"'{res}'\nПо контексту выше, ответь кратко: '{question}'"
    #    print(prompt)
    #    return generate(prompt)
    else:
        return 'None'


def get_bert_vector(text: str) -> list:
    """
    :param text:
    :return:
    """
    req_json = {'search_string': text}
    response_json = requests.post(URL_VECTORISATION, json=req_json)
    bert_vector = response_json.json()['vector']

    return bert_vector


def get_chunks(question: str, model_score) -> list[str]:
    """
    Find most relevant chunks from finder_documents
    - Parameters:
        1. question (str): question text
    - Return:
        1. chunks (list[str]): list of chunks texts
    """
    c = Connection()
    conn, curr = c.conn, c.curr
    curr.execute(f'select * from finder_documents;')
    cursor_columns = [col.name for col in curr.description]
    data = pd.DataFrame(curr.fetchall(), columns=cursor_columns)

    question_vect = eval(get_bert_vector(question))
    data['cos_sim'] = data['vector'].apply(
        lambda x: cos_similarity(question_vect, eval(str(x))).item())
    # lambda x: cosine_similarity(np.array(question_vect).reshape(1,-1), np.array(eval(x)).reshape(1, -1))[0][0])
    data = data.sort_values(by='cos_sim', ascending=False).head(20)  # ['message'])
    data['score'] = data['message'].apply(lambda x: model_score.predict([(question, x)]))
    return list(data.sort_values(by='score', ascending=False).head(5)['message'])


def script_chat(question: str, conversation: Conversation) -> Response:
    """
    """
    conversation.add_user_message(question)
    prompt = conversation.get_prompt()
    output = generate(prompt)
    return Response(question=prompt, answer=output, is_talking_question=1)


def script_pomoshnik(question: str, conversation: Conversation, model_score) -> Response:
    """
    """
    chunks = ' '.join(get_chunks(question, model_score))
    prompt = f"'{chunks}'\nПо контексту выше, ответь кратко: '{question}'"
    answer = generate(prompt)

    prompt = f"'{answer}\n{chunks}'\nПо контексту выше, ответь кратко: '{question}'"
    answer = generate(prompt)

    if check_answer_for_pnzo(answer) or len(answer) == 0:
        conversation.add_user_message(question)
        prompt = conversation.get_prompt()
        output = generate(prompt)
        return Response(question=prompt,
                        answer=PNZO_ANSWER + '\n\n' + 'Возможно вам подойдет ответ: ' + output.split('bot')[0],
                        is_talking_question=0)
    else:
        return Response(question=prompt, answer=answer, is_talking_question=0)


app = FastAPI()


@serve.deployment(ray_actor_options={"num_cpus": NUM_CPUS}, route_prefix="/")
@serve.ingress(app)
class FastAPIDeployment:
    def __init__(self):
        self.model_clf = pickle.load(open("model_script_clf.pickle", "rb"))
        self.model_score = CrossEncoder('cross-encoder/stsb-roberta-large')

    @app.post("/get_answer_for_bot")
    async def get_answer(self, request: Request) -> Response:
        conversation = Conversation()
        chat = eval(request.chat)
        for message in chat[:-1]:
            if message['role'] == 'user':
                conversation.add_user_message(message['text'])
            elif message['role'] == 'bot':
                conversation.add_bot_message(message['text'])
        question = chat[-1]['text']

        # Check for questions_answers
        qa_answer = get_answer_from_qa(question)
        if qa_answer != 'None':
            chunks = ' '.join(get_chunks(question, self.model_score))
            prompt = f"'{chunks}'\nПо контексту выше, ответь кратко: '{question}'"
            conversation.add_user_message(prompt)
            return Response(question=prompt, answer=qa_answer, is_talking_question=0)

        if self.model_clf.predict([question]):
            return script_chat(question, conversation)
        else:
            return script_pomoshnik(question, conversation, self.model_score)

    @app.get("/health", status_code=status.HTTP_200_OK)
    async def healthcheck(self):
        return {"status": "ok"}


serve.run(FastAPIDeployment.bind(), host='0.0.0.0', port=8089)

while True:
    time.sleep(1)