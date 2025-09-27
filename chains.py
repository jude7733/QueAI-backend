from datetime import datetime
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers.pydantic import PydanticOutputParser
import pprint
from utils import load_system_prompt
from schema import AnswerWithCritique, ReviseAnswer
import dotenv

dotenv.load_dotenv()

system_prompt = load_system_prompt()

chat_prompt_template = ChatPromptTemplate(
    [
        (
            "system",
            """{system_prompt}
current time: {time}

1. {first_instruction}
2. Reflection and critique your answer. Be severe to maximize improvements.
3. After the reflection, list 1-3 search queries separately for researching improvements. Do not include them inside the reflection.
""",
        ),
        MessagesPlaceholder(variable_name="messages"),
        ("system", "Answer the user's question above using the required format."),
    ]
).partial(system_prompt=system_prompt, time=datetime.now().isoformat())

first_prompt_template = chat_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

Parser = PydanticOutputParser(pydantic_object=AnswerWithCritique)

responder_chain = first_prompt_template | llm.bind_tools(
    tools=[AnswerWithCritique], tool_choice="AnswerWithCritique"
)

revise_instructions = """Revise your previous answer using the new information.
- You should Ese the previous critique to add important information to your answer.
- You MUST include numerical citations in your revised answer to ensure it can be verified.
- Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
- [1] https://example.com
- [2] https://example.com
- You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""


revisor_chain = chat_prompt_template.partial(
    first_instruction=revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
