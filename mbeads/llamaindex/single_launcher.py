import dotenv
dotenv.load_dotenv() # our .env file defines OPENAI_API_KEY
from llama_agents import (
    AgentService,
    ControlPlaneServer,
    SimpleMessageQueue,
    AgentOrchestrator,
    LocalLauncher,
)
from llama_index.core.agent import FunctionCallingAgentWorker
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI
import logging

# turn on logging so we can see the system working
logging.getLogger("llama_agents").setLevel(logging.INFO)


import os
llm = OpenAI()


# Set up the message queue and control plane
message_queue = SimpleMessageQueue()

control_plane = ControlPlaneServer(
    message_queue=message_queue,
    orchestrator=AgentOrchestrator(llm=llm),
    port=37000
)

# create a tool
def get_the_secret_fact() -> str:
    """Returns the secret fact."""
    return "The secret fact is: A baby llama is called a 'Cria'."

tool = FunctionTool.from_defaults(fn=get_the_secret_fact)

# Define an agent
worker = FunctionCallingAgentWorker.from_tools([tool], llm=OpenAI())
agent = worker.as_agent()

# Create an agent service
agent_service = AgentService(
    agent=agent,
    message_queue=message_queue,
    # description="Useful for getting the secret fact.",
    # service_name="secret_fact_agent",
    description="General purpose assistant",
    service_name="assistant",
)

launcher = LocalLauncher(
    [agent_service],
    control_plane,
    message_queue,
)

# Run a single query through the system
result = launcher.launch_single("What's the secret fact?")
#result = launcher.launch_single("What's the secret fact?")

# from llama_index.core.llms import ChatMessage

# messages = [
#     ChatMessage(
#         role="user", content="What's the secret fact?"
#     )
# ]
# result = await launcher.alaunch_single(messages[0])
print(result)
