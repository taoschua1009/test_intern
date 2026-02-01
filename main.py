import os
import sys
import json
import time
import logging

from core.llm import LLMService
from core.memory import MemoryManager
from core.pipeline import QueryPipeline
from utils.logger import setup_logger


DATA_DIR = "data"
CONV_FILE = "input/long_conversation.jsonl"
QUERY_FILE = "input/ambiguous_queries.jsonl"


logger = setup_logger("DemoRunner")

class DualLogger(object):
    """Ghi output ra cả màn hình và file text"""
    def __init__(self, filepath):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.log = open(filepath, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def save_json_result(filename, data):
    filepath = os.path.join(DATA_DIR, filename)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n[System] Saved result to: {filepath}")

def load_data_smart(filepath):
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        return []

    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            first_char = f.read(1)
            f.seek(0)
            
            if first_char == '[':
                data = json.load(f)
            else:
                for line in f:
                    if line.strip():
                        data.append(json.loads(line))
        
        logger.info(f"Loaded {len(data)} items from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error reading file {filepath}: {str(e)}")
        return []



def run_demo():
    print("=== STARTING CHAT ASSISTANT BACKEND DEMO ===\n")
    
    llm = LLMService(model="gpt-4o-mini")
    
    
    memory_path = os.path.join(DATA_DIR, "memory/memory_store.json")
    memory = MemoryManager(
        llm_service=llm, 
        token_threshold=60, 
        persist_path=memory_path
    )
    
    if os.path.exists(memory_path):
        os.remove(memory_path)
        print("[System] Cleared old memory state.")

    pipeline = QueryPipeline(llm_service=llm, memory_manager=memory)

    # --- FLOW 1: SESSION MEMORY TRIGGER ---
    print("\n>>> DEMO FLOW 1: Session Memory & Summarization Trigger")
    
    conv_path = os.path.join(DATA_DIR, CONV_FILE)
    messages = load_data_smart(conv_path)
    
    if messages:
        for i, msg in enumerate(messages):
            role = msg.get('role', 'user') 
            content = msg.get('content', '')
            
            if not content: continue

            display_role = "User" if role == "user" else "Assistant"
            print(f"[{i+1}] [{display_role}]: {content[:60]}...")
            
            memory.add_message(role, content)
            time.sleep(0.05)
            
        summary_dict = memory.export_summary_output()
        print("\n[Result] Final Session Summary:")
        print(json.dumps(summary_dict, indent=2))
        save_json_result("output/output_session_summary.json", summary_dict)
    else:
        print(f"[WARNING] Skipping Flow 1 because {CONV_FILE} is empty or missing.")

    # --- FLOW 2: AMBIGUOUS QUERY HANDLING ---
    print("\n\n>>> DEMO FLOW 2: Ambiguous Query Understanding")
    
    query_path = os.path.join(DATA_DIR, QUERY_FILE)
    queries = load_data_smart(query_path)
    flow2_results = []

    if queries:
        for item in queries:
            if isinstance(item, dict):
                q = item.get('query')
            elif isinstance(item, str):
                q = item
            else:
                continue

            if not q: continue

            print(f"\nUser Query: '{q}'")
            
            # Run Pipeline
            output = pipeline.process_query(q)
            
            print(f" -> Is Ambiguous: {output.is_ambiguous}")
            if output.is_ambiguous:
                 print(f" -> Rewritten: {output.rewritten_query}")
            
            print(f" -> Memory Keys Needed: {output.needed_context_from_memory}") 
            
            if output.clarifying_questions:
                print(f" -> Clarifying Questions: {output.clarifying_questions}")
            
            # [FIXED] Đã thêm phần in Final Augmented Context
            print(f" -> Final Augmented Context: {output.final_augmented_context}")
            
            flow2_results.append(output.model_dump())
        
        save_json_result("output/output_query_analysis.json", flow2_results)
    else:
         print(f"[WARNING] Skipping Flow 2 because {QUERY_FILE} is empty or missing.")

if __name__ == "__main__":
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    log_file = os.path.join(DATA_DIR, "output/demo_log.txt")
    sys.stdout = DualLogger(log_file)
    
    try:
        run_demo()
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()