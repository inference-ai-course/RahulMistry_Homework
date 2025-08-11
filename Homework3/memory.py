from collections import defaultdict, deque
from typing import Dict, Deque, List, Literal, TypedDict

Role = Literal["user", "assistant"]

class Turn(TypedDict):
    role: Role
    text: str

class MemoryStore:
    def __init__(self, max_turns: int = 5):
        self.max_turns = max_turns
        self._store: Dict[str, Deque[Turn]] = defaultdict(lambda: deque(maxlen=max_turns*2)) 

    def add(self, session_id: str, role: Role, text: str):
        self._store[session_id].append({"role": role, "text": text})

    def get_recent(self, session_id: str) -> List[Turn]:
        return list(self._store[session_id])[-self.max_turns*2:]

    def clear(self, session_id: str):
        if session_id in self._store:
            self._store[session_id].clear()

memory = MemoryStore()
