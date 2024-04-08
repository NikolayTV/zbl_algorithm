from llm_api_calls import send_message_to_gemini_async, send_message_open_router
import json, copy


def convert_gemini_history_to_open_router(gemini_history):
    openrouter_message_history = copy.deepcopy(gemini_history)
    for x in openrouter_message_history:
        # if x['role'] == 'assistant':
        #     x['role'] = 'model'

        if x['role'] == 'model':
            x['role'] = 'assistant'
        if isinstance(x.get('parts'), list):
            x['content'] = x['parts'][0]
            del x['parts']
    return openrouter_message_history


simulate_user_response = """
Вы ИИ алгоритм который отвечает на вопросы, придумывая ответы, для того чтобы просимулировать диалог. Ваши ответы не отличимы от ответа человека, вы овечаете коротко и по делу.
У вас всегда должен быть ответ на вопрос для успешной симуляции. Контекст: доктор задает вам вопросы о здоровье, чтобы составить вам правильный распорядок дня.
Вот вопрос от врача:
<doctor_question>
{$DOCTOR_QUESTION}
</doctor_question>
Ответ:
"""


base_end_dialog_algorithm = """
Вы тактичный и вежливый ИИ-ассистент которого зовут Аика, вы помогаете ученику самостоятельно собрать алгоритм по распорядку дня для улучшения здоровья. 
Вот карточка ученика:
<student_card>
{$STUDENT_CARD}
</student_card>

Состояние диалога: Вы уже в конце разговора. Вам необходимо вежливо закончить этап сбора анамнеза у ученика, и перейти к этапу сбора алгоритма.
Если есть незаполненные поля "values" в карточке ученика <student_card>, то вежливо задайте уточняющие вопросы касающиеся незаполненных полей.
Если же все заполнено то напишите кодовое слово FINISH и мы перейдем к следующему этапу.
"""


base_continue_dialog_algorithm = """Вы тактичный и вежливый ИИ-ассистент которого зовут Аика, вы помогаете ученику самостоятельно собрать алгоритм по распорядку дня для улучшения здоровья. 
Вам по очереди дана json карточка ученика вместе с инструкцииями. Вам необходимо построить диалог таким образом, чтобы заполнить все обязательные поля в карточке. 
Вместо того, чтобы давать готовые решения, старайтесь использовать сократовский метод, задавая наводящие вопросы, чтобы помочь ученику самостоятельно создать свой собственный алгоритм дня.
Начните с первого вопроса про имя, дождись ответа пользователя, и далее следуйте по плану, задавая вопросы по одному, чтобы заполнить все незаполненные поля в карточке - спрашивай только 1 вопрос в одном сообщении. 
Пиши только сообщения пользователю, как будто бы ты общаешься с ним через мессенджер. Будь вежливой, счастливой и остроумной. Не забывай уместно благодорить ученика за ответы.
Если же все заполнено то напишите кодовое слово FINISH и мы перейдем к следующему этапу.

Вот заполненная поля в карточке ученика:
<student_card>
{$STUDENT_CARD}
</student_card>

Вот незаполненные поля в карточке ученика (задавать вопросы по ней последовательно):
<student_card_unfilled>
{$STUDENT_CARD_UNFILLED}
</student_card_unfilled>

Состояние диалога: вы в самом разгаре диалога. Переходите к уточняющим вопросам для заполнения карточки
"""

base_start_dialog_algorithm = """<SYSTEM_MESSAGE>Вы тактичный и вежливый ИИ-ассистент которого зовут Аика, вы помогаете ученику самостоятельно собрать алгоритм по распорядку дня для улучшения здоровья.
Вам дана json карточка ученика вместе с инструкцииями. Вам необходимо построить диалог таким образом, чтобы заполнить все поля в карточке по очереди задавая вопросы.
Вместо того, чтобы давать готовые решения, старайтесь использовать сократовский метод, задавая наводящие вопросы, чтобы помочь ученику самостоятельно создать свой собственный алгоритм дня.
Начните с первого вопроса про имя и возраст и далее следуйте по плану, задавая вопросы по одному.
Пиши только сообщения пользователю, как будто бы ты общаешься с ним через мессенджер. Будь вежливой, счастливой и остроумной. Не забывай уместно благодорить ученика за ответы.
Если все поля заполнены, то напишите кодовое слово FINISH и мы перейдем к следующему этапу.

Вот карточка ученика (задавай все вопросы по ней последовательно в том же порядке, в котором они тут стоят):
<student_card>
{$STUDENT_CARD}
</student_card>

Состояние диалога: самое начало. Начните с приветствия. Затем спросите промокод. Если промокода нет, или пользователь хочет завершить диалог, то вежливо попрощайтесь с пользователем и напишите кодовое слово TERMINATE.
</SYSTEM_MESSAGE>
"""


student_card_template = {
  "description of card": {"value": "карточка о здоровье, хранических заболеваниях, выбранном лечении"},
  "Имя и возраст": {"value": ""},
  "Цели по здоровью": {"value": ""},
  "Промокод": {"value": "", "instruction": "Спросите промокод, который он получил в девятом уроке. Он открывает доступ к формированию алгоритма. Без промокода доступ к алгоритму будет закрыт и Аика выйдет из этого сценария"},
  "Впечатления о курсе": {"value": ""},
  "Артериальное давление": {"value": "", "instruction": "Спросить про точное значение артериального давления, а также какой тип соответствует его давлению: Гипотоник (менее 110/70), нормотоник (110/70 - 130/90) или гипертоник (более 130/90)."},
  "Тип сокращения желчного пузыря": {"value": "", "instruction": "Спросить, сделано ли УЗИ и какой тип сокращения желчного пузыря. Возможные варианты: нормокинетический, гипо, гипер, неизвестно, желчный удален."},
  "Трава желчегонная": {"value": "", "instruction": "Спросить какую он подобрал желчегонную траву. Убедиться, что выбрана трава соответствует типу артериального давления. Гиперторику подходят только кукурузные рыльца которые уменьшают давление. Гипотонику подходит только пижма которая увеличивает давление, а нормотонику подходит тысячелистник который не изменяет давление."},
  "Желчегонная гимнастика": {"value": "", "instruction": "Спросить о наличии камней в желчном пузыре, если их нет, то предложить выполнять желчегонную гимнастику. Урок по ссылке: https://goo.su/WlFpAPi"},
  "Уровень витамина Д в организме": {"value": "", "instruction": "Спросить сдавал ли он анализы и в каких количествах планирует принимать."},
  "Доза витамина Д": {"value": "", "instruction": "Спросить в каких количествах планирует принимать. Если он не уверен, то посоветовать обратиться к Куратору Марии - ник в телеграме @mariasmirnova03. Стандартная доза 4000 МЕ"},
  "Время принятия лимфатического душа": {"value": "", "instruction": "Предложить технику лимфатического душа, спросить предпочтительное время принятия душа (утро или вечер)."},
  "Результаты копрограммы": {"value": "", "instruction": "Спросить о результатах копрограммы, если они имеются."},
  "Хронические заболевания": {"value": "", "instruction": "Наличие заболеваний или противопоказаний"}
}


extract_student_health_data_prompt = """Вам представлен диалог между ИИ-ассистентом с именем Model и Пользователем User. 
Внимательно прочитайте его, и поймите где спрашивает Model, а где отвечает User.
Ваша задача - извлечь релевантную информацию о юзере из фрагмента диалога между моделью и юзером.
Особо важно извлечь те поля, которые представлены в карточке клиента. Кроме этого, нужно извлекать другую важную информацию о юзере, если он ее предоставляет.

Формат диалога:
Model: [Вопрос ассистента]
User: [Ответ пользователя]

Схема карточки здоровья ученика, поле "value" в которой необходимо заполнить:
<student_card>
{$STUDENT_CARD}
</student_card>

Извлеките и представьте информацию в формате питоновского словаря. 
Пример:
```json
{"Артериальное давление": {"value": "нормотоник 120/70"}}
```

Фрагмент диалога:
<dialog>
{$DIALOG}
</dialog>
"""




async def generate_readable_history_from_end(messages, num_pairs):
    """
    Generates a readable history from the end of a list of messages, ensuring that the history
    starts with a message from the model and ends with a message from the user, from the last messages.
    It includes a specified number of model-user message pairs from the end.

    Parameters:
    - messages (list of dicts): The messages in the conversation.
    - num_pairs (int): The number of model-user pairs to include in the output, counting from the end.

    Returns:
    - str: A string representing the readable history.
    """

    # Reverse the list to start from the end
    messages_reversed = list(reversed(messages))

    # Filter messages to get only those with 'model' and 'user' roles
    filtered_messages = [msg for msg in messages_reversed if msg['role'] in ['model', 'user']]

    # Ensure the last message (first in the reversed list) is from the 'user'
    if filtered_messages and filtered_messages[0]['role'] != 'user':
        # If the last message is not from 'user', try to adjust by removing the first message in the reversed list
        filtered_messages = filtered_messages[1:]

    # Limit the number of pairs, considering we're working in reverse
    limited_messages = []
    for i in range(0, min(len(filtered_messages), num_pairs * 2), 2):
        # Check if there's a subsequent (in reverse) model message to form a complete pair
        if i + 1 < len(filtered_messages) and filtered_messages[i]['role'] == 'user' and filtered_messages[i + 1]['role'] == 'model':
            limited_messages.extend(filtered_messages[i:i+2])

    # Generate the readable history from the limited messages, then reverse it to correct the order
    readable_history = '\n'.join([f"{item['role'].title()}: {' '.join(item['parts'])}" for item in reversed(limited_messages)])
    
    return readable_history


async def send_message(prompted_message, history, usage_history=[], rate_limiter=None, provider='gemini'):
    history.append({"role": "user", "parts": [prompted_message]})

    if provider == 'gemini':
        response = await send_message_to_gemini_async(history, rate_limiter=rate_limiter, generation_params={"temperature": 0.2, "top_k":3})

    elif provider == 'router_sonnet':
        openrouter_history = convert_gemini_history_to_open_router(history)
        response = await send_message_open_router(openrouter_history, rate_limiter=None, model="anthropic/claude-3-sonnet")
    

    input_tokens = response.get('input_tokens')
    output_tokens = response.get('output_tokens')
    answer_text = response.get('text_response')

    history.append({"role": "model", "parts": [answer_text]})
    if len(usage_history) > 0:
        conversation_tokens = usage_history[-1].get("conversation_tokens", 0) + input_tokens + output_tokens
    else:
        conversation_tokens = input_tokens + output_tokens

    usage_history.append({"input_tokens":input_tokens, "output_tokens":output_tokens, "conversation_tokens":conversation_tokens})

    print(usage_history[-1])
    print('\n\nUSER:\n', prompted_message)
    print('\n\MODEL:\n', answer_text)
    return history, usage_history


async def add_to_student_card(readable_history, student_card, extraction_history=[], usage_history=[], rate_limiter=None):

    extract_student_health_data_prompted = extract_student_health_data_prompt.replace("$DIALOG", readable_history).replace("$STUDENT_CARD", str(student_card))
    extraction_history = [{"role": "user", "parts": [extract_student_health_data_prompted]}]
    response = await send_message_to_gemini_async(extraction_history, rate_limiter=rate_limiter, generation_params={"temperature": 0, "top_k":1})

    input_tokens = response.get('input_tokens')
    output_tokens = response.get('output_tokens')
    answer_text = response.get('text_response')

    try:
        new_info = json.loads(answer_text.strip('```json').strip('```'))
        student_card.update(new_info)
        print('\n\n NEW INFO', new_info)
    except:
        print('\n\n NEW INFO не найдена', response.get("text_response"))

    if len(usage_history) > 0:
        conversation_tokens = usage_history[-1].get("conversation_tokens", 0) + input_tokens + output_tokens
    else:
        conversation_tokens = input_tokens + output_tokens
    usage_history.append({"input_tokens":input_tokens, "output_tokens":output_tokens, "conversation_tokens":conversation_tokens})
    print(usage_history[-1])
    return {"response":response, "usage_history":usage_history, "student_card":student_card}


def update_message_history_with_system_message(message_history, system_message="<SYSTEM_MESSAGE>"):
    """
    Updates the message history to start from the point including and following
    a specific system message.

    Parameters:
    - message_history (list of dicts): The original message history.
    - system_message (str): The specific system message to find.

    Returns:
    - list of dicts: Updated message history starting from the system message.
    """

    # Find the index of the message that contains the system_message
    for index, message in enumerate(message_history):
        if system_message in message['parts'][0]:  # Assuming system message is always in the first part
            return message_history[index:]  # Return the history from this message onwards

    # If the system message is not found, return the original history
    return message_history
