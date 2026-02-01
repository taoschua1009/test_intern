from typing import List, Optional, Set
from pydantic import BaseModel, Field, field_validator

class ChatMessage(BaseModel):
    role: str
    content: str

class UserProfile(BaseModel):
    prefs: List[str] = Field(default_factory=list, description="User preferences extracted from chat")
    constraints: List[str] = Field(default_factory=list, description="Constraints or limitations mentioned by user")

class MessageRange(BaseModel):
    from_index: int = Field(description="Index of the first summarized message", alias="from")
    to_index: int = Field(description="Index of the last summarized message", alias="to")
    model_config = {"populate_by_name": True}

class SessionSummary(BaseModel):
    user_profile: UserProfile
    key_facts: List[str] = Field(default_factory=list, description="Important facts extracted")
    decisions: List[str] = Field(default_factory=list, description="Decisions made during conversation")
    open_questions: List[str] = Field(default_factory=list, description="Unresolved questions")
    todos: List[str] = Field(default_factory=list, description="Action items")

class SummaryOutput(BaseModel):
    session_summary: SessionSummary
    message_range_summarized: MessageRange

# --- HELPERS CHO VALIDATION ---
def get_allowed_keys_info():
    """
    Trả về whitelist và mapping cho validator.
    Ví dụ mapping: 'constraints' -> 'user_profile.constraints'
    """
    allowed = set(SessionSummary.model_fields.keys())
    mapping = {}
    
    # Map các trường cấp 1
    for k in allowed:
        mapping[k] = k
        
    # Map các trường con của user_profile
    for k in UserProfile.model_fields.keys():
        full_key = f"user_profile.{k}"
        allowed.add(full_key)
        mapping[k] = full_key # Cho phép LLM trả về 'constraints' thay vì full path
        mapping[full_key] = full_key
        
    return allowed, mapping

_ALLOWED_KEYS, _KEY_MAPPING = get_allowed_keys_info()

class QueryUnderstandingOutput(BaseModel):
    original_query: str
    is_ambiguous: bool = Field(description="True if query needs clarification or context")
    rewritten_query: str = Field(description="Self-contained query with context resolved")
    needed_context_from_memory: List[str] = Field(description="Keys needed from memory (e.g. 'key_facts', 'user_profile.constraints')")
    clarifying_questions: List[str] = Field(description="Questions to ask user if ambiguous", max_items=3)
    final_augmented_context: str = Field(description="The context text block to be used for the final answer")
    
    @field_validator("needed_context_from_memory", mode='before')
    @classmethod
    def validate_needed_context_keys(cls, v: List[str]):
        if not v:
            return []
            
        cleaned = []
        for item in v:
            item = item.strip()
            
            # 1. Xử lý tiền tố thừa
            if item.startswith("session_summary."):
                item = item.replace("session_summary.", "")
            
            # 2. Map từ khóa ngắn sang từ khóa đầy đủ (Smart Mapping)
            # Ví dụ: "constraints" -> "user_profile.constraints"
            mapped_key = _KEY_MAPPING.get(item)
            
            if mapped_key:
                cleaned.append(mapped_key)
            else:
                # [DEBUG LOG] Uncomment dòng dưới nếu muốn soi lỗi
                # print(f"DEBUG: Rejected key '{item}' - Not in allowed list")
                pass
                
        # Loại bỏ trùng lặp
        return list(set(cleaned))