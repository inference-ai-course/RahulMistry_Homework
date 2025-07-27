from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama



prompt = PromptTemplate.from_template("What is the capital of {topic}?")


model = ChatOllama(model="llama2") 


chain = (
    {"topic": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)


result = chain.invoke("Germany")


print("User prompt: 'What is the capital of Germany?'")
print("Model answer:", result)
