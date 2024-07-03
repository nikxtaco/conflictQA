import sys
import os
import asyncio
import openai
from openai import OpenAI
import time

from dotenv import load_dotenv
import os
load_dotenv()

# KEY_INDEX = 0
# KEY_POOL = [
#     os.getenv("OPENAI_API_KEY")

openai.api_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(
        # This is the default and can be omitted
        api_key=os.environ.get("OPENAI_API_KEY"),
    )

async def catch_openai_api_error(prompt_input: list):
    global KEY_INDEX
    error = sys.exc_info()[0]
    if error == openai.RateLimitError:
        print('Rate limit error, waiting for 40 second...')
        await asyncio.sleep(40)
    elif error == openai.APIError:
        print('API error, waiting for 1 second...')
        await asyncio.sleep(1)
    elif error ==  openai.Timeout:
        print('Timeout error, waiting for 1 second...')
        await asyncio.sleep(1)
    elif error ==  openai.ServiceUnavailableError:
        print('Service unavailable error, waiting for 3 second...')
        await asyncio.sleep(3)
    elif error ==  openai.APIConnectionError:
        print('API Connection error, waiting for 40 second...')
        await asyncio.sleep(40)
    elif error ==  openai.InvalidRequestError:
        print('Invalid Request Error, trying by pruning the input prompt...')
    
    # if error == openai.BadRequestError: # Edited: Replaced Invalid with Bad
    #     # something is wrong: e.g. prompt too long
    #     print(f"InvalidRequestError\nPrompt:\n\n{prompt_input}\n\n")
    #     assert False
    # elif error == openai.RateLimitError:
    #     KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        
    #     print("RateLimitError", openai.api_key)
    # elif error == openai.APIError:
    #     KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        
    #     print("APIError", openai.api_key)
    # elif error == openai.AuthenticationError:
    #     KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        
    #     print("AuthenticationError", openai.api_key)
    # elif error == TimeoutError:
    #     KEY_INDEX = (KEY_INDEX + 1) % len(KEY_POOL)
        
    #     print("TimeoutError, retrying...")
    else:
        print("API error:", error)


def openai_unit_price(model_name, token_type="prompt"):
    if 'gpt-4' in model_name:
        if token_type == "prompt":
            unit = 0.03
        elif token_type == "completion":
            unit = 0.06
        else:
            raise ValueError("Unknown type")
    elif 'gpt-3.5-turbo' in model_name:
        unit = 0.002
    elif 'davinci' in model_name:
        unit = 0.02
    elif 'curie' in model_name:
        unit = 0.002
    elif 'babbage' in model_name:
        unit = 0.0005
    elif 'ada' in model_name:
        unit = 0.0004
    else:
        unit = -1
    return unit


def calc_cost_w_tokens(total_tokens: int, model_name: str):
    unit = openai_unit_price(model_name, token_type="completion")
    return round(unit * total_tokens / 1000, 4)


def calc_cost_w_prompt(total_tokens: int, model_name: str):
    # 750 words == 1000 tokens
    unit = openai_unit_price(model_name)
    return round(unit * total_tokens / 1000, 4)

async def prompt_chatgpt(system_input, user_input, temperature, save_path, index, history=[], model_name='gpt-3.5-turbo'):
    '''
    :param system_input: "You are a helpful assistant/translator."
    :param user_input: you texts here
    :param history: ends with assistant output.
                    e.g. [{"role": "system", "content": xxx},
                          {"role": "user": "content": xxx},
                          {"role": "assistant", "content": "xxx"}]
    return: assistant_output, (updated) history, money cost
    '''
    
    if len(history) == 0:
        history = [{"role": "system", "content": system_input}]
    history.append({"role": "user", "content": user_input})
    print(openai.api_key)
    while True:
        try:
            completion = client.chat.completions.create(model=model_name, seed=42, messages=history)
            if completion is None:
                raise TimeoutError
            break
        # except:
            # catch_openai_api_error(user_input)
        except openai.RateLimitError:
            print('Rate limit error, waiting for 40 second...')
            await asyncio.sleep(40)
        except openai.APIError:
            print('API error, waiting for 1 second...')
            await asyncio.sleep(1)
        # except openai.Timeout:
        #     print('Timeout error, waiting for 1 second...')
        #     await asyncio.sleep(1)
        # except openai.ServiceUnavailableError:
        #     print('Service unavailable error, waiting for 3 second...')
        #     await asyncio.sleep(3)
        except openai.APIConnectionError:
            print('API Connection error, waiting for 40 second...')
            await asyncio.sleep(40)
        # except openai.InvalidRequestError:
        #     print('Invalid Request Error, trying by pruning the input prompt...')
        except openai.BadRequestError:
            print('Invalid Request Error, trying by pruning the input prompt...')
            # reduce the evidence length in 'user' message
            # if messages[1]['role'] != 'user':
            #     raise ValueError("Input does not have 'user' message")

            time.sleep(1)

    print(completion)
    assistant_output = completion.choices[0].message.content
    history.append({"role": "assistant", "content": assistant_output})
    total_prompt_tokens = completion.usage.prompt_tokens
    total_completion_tokens = completion.usage.completion_tokens

    with open(save_path, 'a+', encoding='utf-8') as f:
        assistant_output = str(index) + "\t" + "\t".join(x for x in assistant_output.split("\n"))
        f.write(assistant_output + '\n')

    print(history)
    return assistant_output, history, calc_cost_w_tokens(total_prompt_tokens, model_name) + calc_cost_w_prompt(
        total_completion_tokens, model_name)
