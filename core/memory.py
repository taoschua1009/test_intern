import json
import tiktoken
import os
from typing import List, Optional
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from schema.definitions import ChatMessage, SummaryOutput, SessionSummary
from core.llm import LLMService
from utils.logger import setup_logger

logger = setup_logger("MemoryManager")

class MemoryManager:
    def __init__(self, llm_service: LLMService, token_threshold: int = 500, persist_path: str = "data/memory_store.json"):
        self.llm = llm_service.get_llm()
        self.token_threshold = token_threshold
        self.persist_path = persist_path # Đường dẫn file lưu trữ

        self.history: List[ChatMessage] = []
        self.summary_output: Optional[SummaryOutput] = None
        self.summarized_count = 0 

        self.tokenizer = tiktoken.get_encoding("cl100k_base")
        self.summary_chain = self.llm.with_structured_output(SummaryOutput)

    # -------- Persistence --------
    def save_state(self):
        """Lưu toàn bộ trạng thái bộ nhớ ra file JSON."""
        try:
            os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
            
            state = {
                "history": [m.model_dump() for m in self.history],
                "summary_output": self.summary_output.model_dump(by_alias=True) if self.summary_output else None,
                "summarized_count": self.summarized_count
            }
            
            with open(self.persist_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2, ensure_ascii=False)
            logger.info(f"Memory state saved to {self.persist_path}")
            
        except Exception as e:
            logger.error(f"Failed to save memory state: {str(e)}")

    def load_state(self):
        """Khôi phục trạng thái bộ nhớ từ file JSON."""
        if not os.path.exists(self.persist_path):
            logger.warning(f"No persistence file found at {self.persist_path}. Starting fresh.")
            return

        try:
            with open(self.persist_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.history = [ChatMessage(**m) for m in state.get("history", [])]
            
            summary_data = state.get("summary_output")
            if summary_data:
                self.summary_output = SummaryOutput(**summary_data)
            
            self.summarized_count = state.get("summarized_count", 0)
            logger.info(f"Memory state loaded. History: {len(self.history)} msgs, Summary exists: {bool(self.summary_output)}")
            
        except (json.JSONDecodeError, ValidationError) as e:
            logger.error(f"Corrupted memory file: {str(e)}. Starting fresh.")

    # -------- Public API --------
    def add_message(self, role: str, content: str):
        self.history.append(ChatMessage(role=role, content=content))
        # Tự động lưu sau mỗi tin nhắn (có thể tối ưu để lưu định kỳ)
        self.save_state() 
        self._check_and_summarize()

    # -------- Token counting --------
    def get_token_count(self) -> int:
        text = "".join([m.content for m in self.history])
        summary_json = self.summary_output.model_dump_json(by_alias=True) if self.summary_output else ""
        return len(self.tokenizer.encode(text + summary_json))

    # -------- Trigger logic --------
    def _check_and_summarize(self):
        curr = self.get_token_count()
        logger.info(f"Token usage: {curr}/{self.token_threshold}")
        
        if curr > self.token_threshold:
            logger.info("Triggering summarization...")
            self._run_summarization()

    # -------- Summarization (With Error Handling) --------
    def _run_summarization(self):
        mid_idx = len(self.history) // 2
        if mid_idx == 0:
            return

        msgs_to_process = self.history[:mid_idx]
        msgs_text = "\n".join([f"{m.role}: {m.content}" for m in msgs_to_process])

        current_summary = (
            self.summary_output.session_summary.model_dump_json(by_alias=True)
            if self.summary_output else "{}"
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a session memory summarizer. Update the JSON summary based on the NEW messages.\n"
             "Return ONLY valid JSON matching the SummaryOutput schema."
            ),
            ("human",
             "Current Summary (JSON): {current_summary}\n\n"
             "New Messages to Absorb:\n{new_messages}\n"
            )
        ])

        chain = prompt | self.summary_chain
        
        try:
            # [Safe Invoke]
            result: SummaryOutput = chain.invoke({
                "current_summary": current_summary,
                "new_messages": msgs_text
            })

            if result:
                processed_count = len(msgs_to_process)
                self.summarized_count += processed_count

                result.message_range_summarized.from_index = 0
                result.message_range_summarized.to_index = self.summarized_count - 1

                self.summary_output = result
                self.history = self.history[mid_idx:]
                
                logger.info(f"Summary Updated Successfully. Range 0 -> {self.summarized_count - 1}")
                self.save_state() # Lưu ngay sau khi tóm tắt
                
        except Exception as e:
            # [Error Handling] Chỉ log lỗi, không crash app, giữ nguyên history để thử lại sau
            logger.error(f"Summarization failed: {str(e)}")
            logger.warning("Skipping summarization this turn.")

    # -------- Context --------
    def get_context(self) -> str:
        ctx = ""
        if self.summary_output:
            ctx += "SUMMARY_OUTPUT (JSON):\n" + self.summary_output.model_dump_json(by_alias=True) + "\n"
        ctx += "\nRECENT MESSAGES:\n" + "\n".join([f"{m.role}: {m.content}" for m in self.history])
        return ctx
    
    

    # -------- Export (Restored for Demo Compatibility) --------
    def export_summary_output(self) -> dict:
        """
        Xuất summary hiện tại ra dạng dict để demo in ra màn hình hoặc lưu file report.
        Sử dụng by_alias=True để giữ key 'from', 'to' đúng theo schema yêu cầu.
        """
        if self.summary_output:
            return self.summary_output.model_dump(by_alias=True)
        return {}