import sys
import os
import asyncio
import openai
import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

prompt_template = '''Question: {search_query}

This is a yes-or-no question. Rewrite this question as a statement, in the affirmative if search_type is yes_statement, and in the negative if search_type is no_statement. The search_type for this statement is {search_type}

Example input:
Question: Is red wine good for the heart?
search_type: yes_statement

Example answer:
Red Wine Is Good for the Heart.

Example input:
Question: Is red wine good for the heart?
search_type: no_statement

Example answer:
Red Wine Is Not Good for the Heart.'''

async def catch_openai_api_error(prompt_input: str):
    error = sys.exc_info()[0]
    if error == openai.RateLimitError:
        print('Rate limit error, waiting for 40 second...')
        await asyncio.sleep(40)
    elif error == openai.APIError:
        print('API error, waiting for 1 second...')
        await asyncio.sleep(1)
    elif error == openai.Timeout:
        print('Timeout error, waiting for 1 second...')
        await asyncio.sleep(1)
    elif error == openai.ServiceUnavailableError:
        print('Service unavailable error, waiting for 3 second...')
        await asyncio.sleep(3)
    elif error == openai.APIConnectionError:
        print('API Connection error, waiting for 40 second...')
        await asyncio.sleep(40)
    elif error == openai.InvalidRequestError:
        print('Invalid Request Error, trying by pruning the input prompt...')
    else:
        print("API error:", error)

async def prompt_chatgpt(system_input, user_input, temperature=0.7, model_name='gpt-3.5-turbo'):
    history = [{"role": "system", "content": system_input}, {"role": "user", "content": user_input}]
    
    while True:
        try:
            completion = client.chat.completions.create(model=model_name, messages=history)
            if completion is None:
                raise TimeoutError
            break
        except openai.RateLimitError:
            print('Rate limit error, waiting for 40 second...')
            await asyncio.sleep(40)
        except openai.APIError:
            print('API error, waiting for 1 second...')
            await asyncio.sleep(1)
        except openai.Timeout:
            print('Timeout error, waiting for 1 second...')
            await asyncio.sleep(1)
        except openai.ServiceUnavailableError:
            print('Service unavailable error, waiting for 3 second...')
            await asyncio.sleep(3)
        except openai.APIConnectionError:
            print('API Connection error, waiting for 40 second...')
            await asyncio.sleep(40)
        except openai.BadRequestError:
            print('Invalid Request Error, trying by pruning the input prompt...')
            time.sleep(1)

    assistant_output = completion.choices[0].message.content
    return assistant_output

def process_search_queries(csv_input_path, csv_output_path):  # Adjusted function parameters
    # Load data from CSV
    df = pd.read_csv(csv_input_path)

    df = df[:4] # Edited
    
    # Initialize new columns
    df['search_engine_input'] = None

    # Process each row
    async def process_row(row):
        try:
            prompt = prompt_template.format(search_query=row['search_query'], search_type=row['search_type'])
            response = await prompt_chatgpt("You are a helpful assistant.", prompt)
            print(response)
            if row['search_type'] == 'yes_statement':
                row['search_engine_input'] = response.split('Yes: ')[-1].strip()
            else:
                row['search_engine_input'] = response.split('No: ')[-1].strip()
        except Exception as e:
            print(f"Error processing row {row.name}: {e}")
            row['search_engine_input'] = None
        return row

    async def process_all_rows(df):
        tasks = [process_row(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)

    loop = asyncio.get_event_loop()
    df = loop.run_until_complete(process_all_rows(df))

    # Save the updated data to CSV
    df.to_csv(csv_output_path, index=False)  # Save to CSV

# Example usage
process_search_queries('pre_search_engine_input.csv', 'post_search_engine_input.csv')
