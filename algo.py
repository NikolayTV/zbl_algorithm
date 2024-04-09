from llm_api_calls import send_message_to_gemini_async, send_message_open_router
import json, copy
from textwrap import dedent


def convert_gemini_history_to_open_router(gemini_history):
    openrouter_message_history = copy.deepcopy(gemini_history)
    for x in openrouter_message_history:
        if x['role'] == 'model':
            x['role'] = 'assistant'
        if isinstance(x.get('parts'), list):
            x['content'] = x['parts'][0]
            del x['parts']
    return openrouter_message_history


def simulate_user_response_prompt():
    return dedent("""\
    <SYSTEM_MESSAGE>
    Вы ИИ алгоритм который поддерживает диалог, каждый раз придумывая вымышленные ответы. 
    Сценарий: вы на приеме у доктора, до этого вы сдали все свои анализы, результаты анализов придумываете на ходу.
    Ваши ответы не отличимы от ответа человека, вы овечаете коротко и по делу.
    Вот вопрос от врача:
    <doctor_question>
    {$DOCTOR_QUESTION}
    </doctor_question>
    </SYSTEM_MESSAGE>
    Ответ:
    """
    )


def base_continue_dialog_algorithm():
    return dedent("""\
    <SYSTEM_MESSAGE>
    Вы вежливый, шутливый и уважительный ИИ-ассистент которого зовут Аика, вы помогаете ученику самостоятельно собрать алгоритм по распорядку дня для улучшения здоровья.
    Вам дана json карточка ученика вместе с инструкцииями. Вам необходимо построить диалог таким образом, чтобы заполнить все поля в карточке по очереди задавая вопросы в том же порядке, в котором поля стоят в карточке.
    Вместо того, чтобы давать готовые решения, старайтесь использовать сократовский метод, задавая наводящие вопросы, чтобы помочь ученику закрепить знания выученные на курсе и самостоятельно создать свой алгоритм дня.
    Пиши только сообщения пользователю, будто ты общаешься с ним через мессенджер.
    Когда все поля будут заполнены, то напишите кодовое слово NEXT_STEP и мы перейдем к следующему этапу.

    Вот карточка ученика:
    <student_card>
    {$STUDENT_CARD}
    </student_card>
                  
    Состояние диалога: вы в процессе диалога. Сразу же переходите к уточняющим вопросам для заполнения карточки
    </SYSTEM_MESSAGE>
    """)


def base_start_dialog_algorithm():
    return dedent("""\
    <SYSTEM_MESSAGE>
    Вы вежливый, шутливый и уважительный ИИ-ассистент которого зовут Аика, ваша задача спрашивать вопросы для заполнения карточки ученика для создания алгоритма по распорядку дня для улучшения здоровья.
    Задавая вопросы, ученик должен самостоятельно формировать основные элементы алгоритма. Подсказки будут даны в карточке, в поле instruction.
    Вам дана json карточка ученика вместе с инструкцииями. Вам необходимо построить диалог таким образом, чтобы заполнить все поля в карточке по очереди задавая вопросы в том же порядке, в котором поля стоят в карточке.
    Вместо того, чтобы давать готовые решения, старайтесь использовать сократовский метод, задавая наводящие вопросы, чтобы помочь ученику закрепить знания выученные на курсе и самостоятельно создать свой алгоритм дня.
    Пишите только сообщения пользователю, будто ты общаешься с ним через мессенджер. Задавайте только вопросы, относящиеся к темам перечисленным в карточке клиента, и не уходите от темы разговора.
    Когда все поля будут заполнены, то напишите кодовое слово NEXT_STEP и мы перейдем к следующему этапу.

    Вот карточка ученика:
    <student_card>
    {$STUDENT_CARD}
    </student_card>
                  
    </SYSTEM_MESSAGE>
    """)


def student_card_template():
    return {
        "description of card": 
            {"value": "карточка о здоровье, хранических заболеваниях, выбранном лечении"},
        "Промокод": 
            {"value": ""},
        "Имя и возраст": 
            {"value": ""},
        "Цели по здоровью": 
            {"value": ""},
        "Артериальное давление": 
            {"value": "", 
            "instruction": "Спросить про точное значение и тип артериального давления. Возможные типы: Гипотоник (менее 110/70), нормотоник (110/70 - 130/90) или гипертоник (более 130/90)."},
        "Тип сокращения желчного пузыря": 
            {"value": "", 
            "instruction": "Спросить, сделано ли УЗИ и какой тип сокращения желчного пузыря. Возможные варианты: нормокинетический, гипо, гипер, неизвестно, желчный удален."},
        "Выбранная трава желчегонная": 
            {"value": "", 
            "instruction": "Спросить какую он подобрал желчегонную траву. Убедиться, что выбрана трава соответствует типу артериального давления. Гиперторику подходят только кукурузные рыльца которые уменьшают давление. Гипотонику подходит только пижма которая увеличивает давление, а нормотонику подходит тысячелистник который не изменяет давление."},
        "Желчегонная гимнастика": 
            {"value": "", 
            "instruction": "Спросить о наличии камней в желчном пузыре, если их нет, то предложить выполнять желчегонную гимнастику. Уточнить когда он буде ее делать. Урок по ссылке: https://goo.su/WlFpAPi"},
        "Витамин Д": 
            {"value": "", 
            "instruction": "Спросить сдавал ли он анализы и в каких количествах планирует принимать. Спросить в каких количествах планирует принимать. Если он не уверен, то посоветовать обратиться к Куратору Марии - ник в телеграме @mariasmirnova03. Стандартная доза 4000 МЕ"},
        "Время принятия лимфатического душа": 
            {"value": "", 
            "instruction": "Предложить технику лимфатического душа, спросить предпочтительное время принятия душа (утро или вечер)."},
        "Результаты копрограммы": 
            {"value": "", 
            "instruction": "Спросить о результатах копрограммы, если они имеются."},
        "Хронические заболевания или противопоказания": 
            {"value": "", 
            "instruction": ""}
        }

def extract_student_health_data_prompt():
    return dedent("""/
    Инструкция:
    Вам представлен фрагмент диалога между ИИ-ассистентом с именем Model и Пользователем User. 
    Внимательно прочитайте его, и поймите где спрашивает Model, а где отвечает User.
    Ваша задача - извлечь релевантную информацию о юзере из фрагмента диалога между моделью и юзером.
    Особо важно извлечь те поля, которые представлены в карточке клиента. Кроме этого, нужно извлекать другую важную информацию о юзере, если он ее предоставляет.
    Выводи только те поля, где ты можешь заполнить значение value. Если пользователь не отвечает на вопрос, то выведи значение "не ответил".
                  
    Формат диалога:
    Model: [Вопрос ассистента]
    User: [Ответ пользователя]

    Схема карточки здоровья ученика, поле "value" в которой необходимо заполнить:
    <student_card>
    {$STUDENT_CARD}
    </student_card>

    Извлеките и представьте информацию в формате питоновского словаря в соответстви с примерами.
    Пример 1:
    ```json
    {"Артериальное давление": {"value": "нормотоник 120/70"}}
    ```

    Пример 2:
    ```json
    {"Выбранная трава желчегонная": {"value": "Тысячелистник"}, "Желчегонная гимнастика": {"value": "Будет делать утром"}, }
    ```

    Фрагмент диалога:
    <dialog>
    {$DIALOG}
    </dialog>

    Ответ:              
    ```json
    """)



async def generate_readable_history_from_end(messages, dialogs_num):
    """
    Generates a readable history from the end of a list of messages, ensuring that the history
    starts with a message from the model and ends with a message from the user, from the last messages.
    It includes a specified number of model-user message pairs from the end.

    Parameters:
    - messages (list of dicts): The messages in the conversation.
    - dialogs_num (int): The number of model-user pairs to include in the output, counting from the end.

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
    for i in range(0, min(len(filtered_messages), dialogs_num * 2), 2):
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
    print('\n\nMODEL:\n', answer_text)
    return history, usage_history



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
