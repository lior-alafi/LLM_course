
from langchain_ollama import ChatOllama
# from langchain.prompts import ChatPromptTemplate
# from langchain.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from dotenv import load_dotenv
from doit_chain import BashCmd
from shell_exec import run_shell
# from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import os
import sys

load_dotenv()

# llm=ChatOllama(model="gemma3:4b")
llm=ChatGoogleGenerativeAI(model="gemini-2.5-flash",google_api_key=os.environ["GOOGLE_API_KEY"])

parser = PydanticOutputParser(pydantic_object=BashCmd)
# format_instructions = parser.get_format_instructions()
prompt=ChatPromptTemplate.from_template(""" 
Your task is to analyze the user's query and determine the appropriate action.
If the user asks to perform a bash command, extract the command and set the intent to "execute_command".
If the user is engaging in conversation (e.g., greetings, jokes, general questions), set the intent to "conversation".
If the query is unclear, ambiguous, or cannot be fulfilled, set the intent to "error".
User Query: {query}
Return the result in the following JSON format:
{format_instructions}
"""
)
partial_prompt = prompt.partial(format_instructions=parser.get_format_instructions())
chain = partial_prompt | llm | parser

def main():
    if len(sys.argv) < 2:
        print('Usage: python doit.py "<your query>"')
        sys.exit(1)
        
    query = sys.argv[1]
    response = chain.invoke({"query": query})
    
    print(f"Intent: {response.intent}")
    if response.intent=="error":
        print(response.error)
    elif response.intent=="conversation":
        print(response.conversation)
    else:
        print(f"Command: {response.command}\n")
        res=run_shell(response.command)
        if res["stdout"]:
            print(res["stdout"])
        if res["stderr"]:
            print("Error Output:", res["stderr"])

if __name__ == "__main__":
    main()