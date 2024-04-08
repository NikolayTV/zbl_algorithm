import os
import google.generativeai as genai
import asyncio
from private import GEMINI_API_KEY, OPENROUTER_API_KEY
from datetime import datetime, timedelta
import requests
import json




class RateLimiter:
    def __init__(self, calls_per_period, period=1.0):
        self.calls_per_period = calls_per_period
        self.period = timedelta(seconds=period)
        self.calls = []

    async def wait(self):
        now = datetime.now()
        
        while self.calls and now - self.calls[0] > self.period:
            self.calls.pop(0)
            
        if len(self.calls) >= self.calls_per_period:
            sleep_time = (self.period - (now - self.calls[0])).total_seconds()
            await asyncio.sleep(sleep_time)
            return await self.wait()

        self.calls.append(datetime.now())



def send_message_to_gemini(user_input):
    genai.configure(api_key=GEMINI_API_KEY)
    # genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
    generation_config = {
        "temperature": 0.9,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 2048,
    }

    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "block_none"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "block_none"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "block_none"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "block_none"},
    ]

    gemini = genai.GenerativeModel(
        model_name="gemini-1.0-pro",  # Или другую модель, например, "models/gemini-1.5-pro"
        generation_config=generation_config,
        safety_settings=safety_settings)

    convo = gemini.start_chat(history=[])
    convo.send_message(user_input)

    gemini_response_text = convo.last.text
    return {"text_response": gemini_response_text, 
            "input_tokens": gemini.count_tokens(user_input), 
            "output_tokens":gemini.count_tokens(gemini_response_text)}
    

import google.generativeai as genai
from private import GEMINI_API_KEY
import asyncio


async def send_message_to_gemini_async(history, rate_limiter=None, attempt=1, max_attempts=10, retry_delay = 1, generation_params={}):
    if rate_limiter is not None: await rate_limiter.wait()

    genai.configure(api_key=GEMINI_API_KEY)
        
    history_gemini = history.copy()
    for h in history_gemini:
        if h['role'] == 'assistant':
            h['role'] = 'model'
        if isinstance(h.get('content'), str):
            h['parts'] = [h['content']]
            del h['content']


    user_input = history_gemini[-1]['parts'][0]
    history_gemini = history_gemini[:-1]

    generation_config = {
        "temperature": 1,
        "top_p": 1,
        "top_k": 5,
        # "max_output_tokens": 2048,
    }

    for key, value in generation_params.items():
        generation_config[key] = value

    # BLOCK_ONLY_HIGH block_none
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_ONLY_HIGH"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_ONLY_HIGH"},
    ]
    try:
        gemini = genai.GenerativeModel(model_name="gemini-1.0-pro",
                                        generation_config=generation_config,
                                        safety_settings=safety_settings)

        convo = gemini.start_chat(history=history_gemini)

        convo.send_message(user_input)

        gemini_response_text = convo.last.text
    except Exception as e:
        print(f'Exception {e}')
        gemini_response_text = 'error in response'


    return {"text_response": gemini_response_text, 
            "input_tokens": gemini.count_tokens(user_input).total_tokens, 
            "output_tokens":gemini.count_tokens(gemini_response_text).total_tokens}




async def send_message_open_router(history, rate_limiter=None, model="anthropic/claude-3-sonnet"):
    if rate_limiter is not None: await rate_limiter.wait()

    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                # "HTTP-Referer": f"{YOUR_SITE_URL}", # Optional, for including your app on openrouter.ai rankings.
                # "X-Title": f"{YOUR_APP_NAME}", # Optional. Shows in rankings on openrouter.ai.
            },
            data=json.dumps({
                # "model": "anthropic/claude-3-opus",
                "model": "anthropic/claude-3-sonnet",
                # "model": "anthropic/claude-3-haiku", # Optional
                "messages": history
            })
            )
        response_text = response.json()['choices'][0]['message']['content']
        input_tokens = response.json()['usage']['prompt_tokens']
        output_tokens = response.json()['usage']['completion_tokens']
    except Exception as e:
        print(f'Exception {e}')
        gemini_response_text = 'error in response'
        input_tokens = 0
        output_tokens = 0

    if response.status_code == 200 or 'error' not in response.json():
        return {"text_response": response_text, 
                "input_tokens": input_tokens, 
                "output_tokens": output_tokens}

    else:
        return {"text_response": f'error {response.status_code}', 
                "input_tokens": response.json()['usage']['prompt_tokens'], 
                "output_tokens": response.json()['usage']['completion_tokens']}
