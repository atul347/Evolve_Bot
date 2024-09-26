from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from finbot.data_ingestion import ingestdata


def generation(vstore):
    retriever = vstore.as_retriever(search_kwargs={"k": 20})

    PRODUCT_BOT_TEMPLATE = """
    Your finance bot is an expert in providing credit card and loan-related advice.
    Ensure your answers are strictly relevant to the query context and do not stray off-topic, greetings are fine. Do not provide any advice or suggestions beyond Evolve credit cards or loans.
    Your responses should be concise and informative.
    Recommend only Evolve credit cards or loans when the user asks for such products, and include the provided link in your recommendation without modifying it.
    Do not suggest any product links when the user is not specifically asking for a credit card or loan.

  


    CONTEXT:
    {context}

    QUESTION: {question}

    YOUR ANSWER:
    
    """
    prompt = ChatPromptTemplate.from_template(PRODUCT_BOT_TEMPLATE)

    llm = ChatOpenAI()

    def format_output(output):
        import re

        # Find all URLs in the output text
        urls = re.findall(r'(https?://\S+)', output)

        # Replace URLs with clickable links
        for url in urls:
            # Ensure we do not include trailing punctuation in the URL
            clean_url = re.match(r'https?://[^\s)]+', url).group(0)
            output = output.replace(clean_url, f'<a href="{clean_url}" target="_blank">{clean_url}</a>')

        return output.replace('\n', '<br>')

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        | format_output
    )

    return chain

if __name__ == '__main__':
    vstore = ingestdata("None")
    chain = generation(vstore)
    response = chain.invoke("card")
    print(response)
