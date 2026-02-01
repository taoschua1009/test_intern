import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

class LLMService:
    
    def __init__(self, model: str = "gpt-4o-mini"): 
        # LangChain tự động lấy OPENAI_API_KEY từ biến môi trường
        self.llm = ChatOpenAI(model=model, temperature=0)

    def get_llm(self):
        return self.llm