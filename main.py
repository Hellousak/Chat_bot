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
    You are Einstein. 
    Answer questions through the lens of Einstein's theories and knowledge.
    You will speak from your perspective as Einstein.
    You will share your personal things from your life even if 
    the user does not ask for it. For example, if the user ask 
    about the theory of relativity, you will share your personal experience, not only
    the theory itself. You will have a sense of humor. Answers should be very short in max 2 sentences."""

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

print("Hi, I´m Albert, how can I help you today?")


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
    gr.Markdown("# Chat with Einstein"
                "\nAsk me anything about physics or my life...")

    chatbot = gr.Chatbot(
        type='messages',
        avatar_images=[None, 'einstein.png'],
        show_label=False
    )

    msg = gr.Textbox(show_label=False, placeholder="Ask Einstein anything...")
    msg.submit(chat, [msg, chatbot], [chatbot, msg], queue=False)

    clear = gr.Button("Clear chat", variant="secondary", size="sm")
    clear.click(clear_chat, outputs=[chatbot, msg], queue=False)

page.launch(share=True, inbrowser=True, server_port=7860)