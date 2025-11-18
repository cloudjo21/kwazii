from rich.pretty import pprint

from agno.memory.v2.db.sqlite import SqliteMemoryDb
from agno.memory.v2.memory import Memory


user_id = "user_12345"  # Example user ID, can be any unique identifier for the user
session_id = "session_67890"  # Example session ID, can be any unique identifier for the session

memory_db = SqliteMemoryDb(table_name="team_memory", db_file="tmp/memory.db")
memory = Memory(db=memory_db, debug_mode=True)

memories = memory.get_user_memories(user_id=user_id)
print("Memories:")
pprint(memories)

# memory.delete_user_memory(
#     user_id=user_id,
#     memory_id="da73555f-896a-4b55-bc58-98b174f0c0eb",
# )
