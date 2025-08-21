from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAI

def get_analysis_chain(llm: OpenAI) -> LLMChain:
    """
    Returns an analysis chain that generates service recommendations.
    """
    prompt_template = """
    You are an expert car mechanic. Based on the following vehicle details, please recommend a list of services and parts to be replaced.

    Vehicle Details:
    - Brand: {brand}
    - Model: {model}
    - Age: {age}
    - Mileage: {mileage}

    Past Service History:
    {history}

    Recommended Services:
    """

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["brand", "model", "age", "mileage", "history"]
    )

    chain = LLMChain(llm=llm, prompt=prompt)
    return chain
