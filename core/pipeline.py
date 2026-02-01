from langchain_core.prompts import ChatPromptTemplate
from schema.definitions import QueryUnderstandingOutput, _ALLOWED_KEYS
from core.llm import LLMService
from core.memory import MemoryManager
from utils.logger import setup_logger

logger = setup_logger("QueryPipeline")

class QueryPipeline:
    def __init__(self, llm_service: LLMService, memory_manager: MemoryManager):
        self.llm = llm_service.get_llm()
        self.memory = memory_manager
        self.parser_llm = self.llm.with_structured_output(QueryUnderstandingOutput)
        self.valid_keys_str = ", ".join([f"'{k}'" for k in _ALLOWED_KEYS])

    def process_query(self, user_query: str) -> QueryUnderstandingOutput:
        context_str = self.memory.get_context()
        
        logger.info(f"Processing query: '{user_query}'")

        prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "You are an intelligent query understanding engine. Analyze the user query based on the CONTEXT.\n"
             "Perform the following steps:\n"
             "1. Detect Ambiguity: Check if the query refers to previous entities (it, that, the system).\n"
             "2. Rewrite Query: If ambiguous, rewrite it to be self-contained.\n"
             "3. Identify Memory Keys: Select relevant keys from this list: [{valid_keys_list}].\n"
             "   - Example: Use 'user_profile.constraints' for limitations, 'key_facts' for tech stack.\n" 
             "4. Augment Context: Construct a 'final_augmented_context' text block summary.\n"
             "5. Clarify: If unsafe or unclear, generate clarifying questions."
            ),
            ("human", "CONTEXT:\n{context}\n\nUSER QUERY: {query}")
        ])

        chain = prompt | self.parser_llm
        
        try:
            result = chain.invoke({
                "context": context_str,
                "query": user_query,
                "valid_keys_list": self.valid_keys_str # Inject danh sách key
            })
            
            # Log kết quả để kiểm tra
            logger.info(f"Analyzed: Ambiguous={result.is_ambiguous}, Keys={result.needed_context_from_memory}")
            return result
            
        except Exception as e:
            logger.error(f"LLM Query Processing failed: {str(e)}")
            return QueryUnderstandingOutput(
                original_query=user_query,
                is_ambiguous=False,
                rewritten_query=user_query,
                needed_context_from_memory=[],
                clarifying_questions=[],
                final_augmented_context=context_str[-1000:]
            )