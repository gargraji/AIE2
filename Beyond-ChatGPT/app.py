# You can find this code for Chainlit python streaming here (https://docs.chainlit.io/concepts/streaming/python)

# OpenAI Chat completion
import os
from openai import AsyncOpenAI  # importing openai for API usage
import chainlit as cl  # importing chainlit for our app
from chainlit.prompt import Prompt, PromptMessage  # importing prompt tools
from chainlit.playground.providers import ChatOpenAI  # importing ChatOpenAI tools
from dotenv import load_dotenv

load_dotenv()

# ChatOpenAI Templates
system_template = """You are a helpful assistant who always speaks in a pleasant tone!
"""

#user_template = """{input}
#Think through your response step by step. """

task_settings = {
    "explanation": {
      "model": "gpt-3.5-turbo",  
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.5,
    },
    "summarization": {
       "model": "gpt-3.5-turbo",
        "temperature": 0.3,
        "max_tokens": 300,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    },
    "creative_writing": {
       "model": "gpt-3.5-turbo",
        "temperature": 0.9,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.5,
    },
    "logical_reasoning": {
       "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 100,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    },
    "style_transformation": {
        "model": "gpt-3.5-turbo",
        "temperature": 0.7,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.5,
    },
}

def detect_task_type(user_input):
    if "explain" in user_input.lower():
        return "explanation"
    elif "summarize" in user_input.lower():
        return "summarization"
    elif "write a story" in user_input.lower() or "imagine" in user_input.lower():
        return "creative_writing"
    elif "how many" in user_input.lower() or "calculate" in user_input.lower():
        return "logical_reasoning"
    elif "rewrite" in user_input.lower() or "formal tone" in user_input.lower():
        return "style_transformation"
    else:
        return "explanation"  # Default to explanation

task_user_templates = {
    "explanation": """{input}
    Explain this concept in simple terms with examples.""",
    "summarization": """{input}
    Summarize the main points in a concise manner.""",
    "creative_writing": """{input}
    Use your imagination to craft a creative and engaging response.""",
    "logical_reasoning": """{input}
    Provide a clear and accurate solution to this problem.""",
    "style_transformation": """{input}
    Rewrite the following text in a professional and formal tone.""",
}

def get_user_template(task_type, user_input):
    return task_user_templates.get(task_type, """{input}
    Think through your response step by step.""").format(input=user_input)

@cl.on_chat_start  # marks a function that will be executed at the start of a user session
async def start_chat():
    settings = {
        "model": "gpt-3.5-turbo",
        "temperature": 0,
        "max_tokens": 500,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0,
    }

    cl.user_session.set("settings", settings)


@cl.on_message
async def main(message: cl.Message):
    # Detect the task type from the user input
    task_type = detect_task_type(message.content)
    # Retrieve the corresponding settings
    settings = task_settings.get(task_type, task_settings["explanation"])
    user_template = get_user_template(task_type, message.content)

    client = AsyncOpenAI()

    print(message.content)

    prompt = Prompt(
        provider=ChatOpenAI.id,
        messages=[
            PromptMessage(
                role="user",
                template=user_template,
                formatted=user_template, #.format(input=message.content)
            )
        ],
        inputs={"input": message.content},
        settings=settings
    )

    print([m.to_openai() for m in prompt.messages])

    msg = cl.Message(content="")

    # Call OpenAI
    async for stream_resp in await client.chat.completions.create(
        messages=[m.to_openai() for m in prompt.messages], stream=True, **settings
    ):
        token = stream_resp.choices[0].delta.content
        if not token:
            token = ""
        await msg.stream_token(token)

    # Update the prompt object with the completion
    prompt.completion = msg.content
    msg.prompt = prompt

    # Send and close the message stream
    await msg.send()
