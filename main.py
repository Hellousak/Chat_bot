from dotenv import load_dotenv
import os
import gradio as gr

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

system_prompt = """
    You are a cat, who is very clever and sarcastic.. 
    You are a great journalist and you are usually writing articles about influencers and lifestyle.
    Your style of writing does not let the reader to stop reading and makes them hungry for more.
    Your name is Doree.
    You are very sarcastic and you like to make fun of people.
    You are very funny and you like to make jokes."""

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.5,
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    (MessagesPlaceholder(variable_name="history")),
    ("user", "{input}")]
)

chain = prompt | llm | StrOutputParser()

print("meow, I´m Doree, what the hell do you want from me?")


def chat(user_input, hist):
    if hist is None:
        hist = []
    langchain_history = []
    for item in hist:
        if item['role'] == 'user':
            langchain_history.append(HumanMessage(content=item['content']))
        elif item['role'] == 'assistant':
            langchain_history.append(AIMessage(content=item['content']))
    response = chain.invoke({"input": user_input, "history": langchain_history})
    hist = hist + [
        {'role': 'user', 'content': user_input},
        {'role': 'assistant', 'content': response}
    ]
    return hist, ""  # První návratová hodnota je historie, druhá vyprázdní textbox

def clear_chat():
    return [], ""  # Vyprázdní chat a textbox

# Gradio interface
page = gr.Blocks(
    title="Chat with Einstein",
    theme=gr.themes.Soft()
)

with page:
    gr.Markdown("# Chat with Doree"
                "\nLet her take you on a journey through the world of influencers and lifestyle. ")

    chatbot = gr.Chatbot(
        type='messages',
        avatar_images=[None, 'doree.png'],
        show_label=False
    )

    msg = gr.Textbox(show_label=False, placeholder="Ask Doree anything...")
    msg.submit(chat, [msg, chatbot], [chatbot, msg], queue=False)

    clear = gr.Button("Clear chat", variant="secondary", size="sm")
    clear.click(clear_chat, outputs=[chatbot, msg], queue=False)

page.launch(share=True, inbrowser=True, server_port=7860)
