#!/usr/bin/env python
# coding: utf-8

## setup required functions

from langchain.prompts import PromptTemplate
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain.schema import AIMessage
import getpass
import os
import PyPDF2


## sign in with api_key = "nvapi-...."

if os.environ.get("NVIDIA_API_KEY", "").startswith("nvapi-"):
    print("Valid NVIDIA_API_KEY already in environment. Delete to reset")
else:
    nvapi_key = getpass.getpass("NVAPI Key (starts with nvapi-): ")
    assert nvapi_key.startswith("nvapi-"), f"{nvapi_key[:5]}... is not a valid key"
    os.environ["NVIDIA_API_KEY"] = nvapi_key


## test if the model works, can be any simple question.
llm = ChatNVIDIA(model="meta/llama2-70b", max_tokens=500) 
result = llm.invoke("What is the weather in Taiwan like in June?")
print(result.content)


## extract the essay from pdf file 

def extract_text_from_pdf(pdf_file_path):
    with open(pdf_file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        essay_text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            essay_text += page.extract_text()
    return essay_text


## evaluation essay prompt

def evaluate_essay_quality(pdf_file_path):
    # Extract text from PDF file
    essay_text = extract_text_from_pdf(pdf_file_path)

    # Set up the language model
    llm = ChatNVIDIA(model="meta/llama2-70b", max_tokens=1000, temperature=0)

    # Define a prompt template for the essay evaluation
    prompt = PromptTemplate(
        input_variables=["essay"],
        template="""
        Please evaluate the quality of the following essay:

        {essay}

        Provide a detailed analysis of the strengths and weaknesses of the essay, and give an overall quality score on a scale of 1 to 10.
        """
    )

    # Generate the essay evaluation using the language model
    result = llm([AIMessage(content=prompt.format(essay=essay_text))])

    return result.content


## Example usage of essay evaluation.
## The evaluation can be saved into pdf file.

pdf_file_path = 'path/to/essay.pdf'
evaluation = evaluate_essay_quality(pdf_file_path)
print(evaluation)


## a case of evaluated by a stricted technical background teacher.
## the rubric of the evaluation can be set according to the teacher's preference.

def evaluate_essay_quality_s(pdf_file_path):
    # Extract text from PDF file
    essay_text = extract_text_from_pdf(pdf_file_path)

    # Set up the language model
    llm = ChatNVIDIA(model="meta/llama2-70b", max_tokens=1000, temperature=0)

    # Define a prompt template for the essay evaluation
    prompt = PromptTemplate(
        input_variables=["essay"],
        template="""
        As a strict technical background teacher, I will provide a thorough evaluation of the following essay:

    {essay}

    The essay will be assessed on the following criteria:
    - Clarity and conciseness of the writing
    - Logical flow and organization of the content
    - Appropriate use of technical terms and concepts
    - Depth of understanding and analysis of the subject matter
    - Overall effectiveness in conveying the key points

    Please note that I will be evaluating this essay with a critical eye and a focus on technical excellence, and give an overall quality score on a scale of 1 to 10.
        """
    )

    # Generate the essay evaluation using the language model
    result_s = llm([AIMessage(content=prompt.format(essay=essay_text))])

    return result_s.content



## Example usage of essay evaluated by a strict technology teacher

evaluation = evaluate_essay_quality_s(pdf_file_path)
print(evaluation)


## Debate with the evaluation result by ask LLM to revise its evaluation
## Evaluation file is saved in pdf

def debate_with_llm(pdf_ev_file_path, human_opinion):
    # Extract text from PDF file
    llm_response = extract_text_from_pdf(pdf_ev_file_path)
    
    # Set up the language model
    llm = ChatNVIDIA(model="meta/llama2-70b", max_tokens=1000, temperature=0)

    # Define prompt template
    prompt = PromptTemplate(
        input_variables=["llm_evaluation", "human_opinion"],
        template="""
        The language model has provided the following evaluation:

        {llm_evaluation}

        As a human teacher has given opinion on the essay:

        {human_opinion}

        The human teacher would like you to revise your evaluation. Please add human teacher's opinion and summary the evaluation in 200 words. Revise and give an overall quality score on a scale of 1 to 10.
        """
    )
    human_response = llm([AIMessage(content=prompt.format(llm_evaluation=llm_response, human_opinion=human_opinion))])
    return human_response.content



## Example usage of debate with the LLM on the evaluation result.

pdf_ev_file_path = 'path/to/evaluation.pdf'

human_opinion = """
As a human teacher, I agree with your opinion. Yet I believe the author demonstrates a good understanding of the topic and the potential of extending the result further. 

Overall, I would rate the essay a 8 out of 10.
"""

human_response = debate_with_llm(pdf_ev_file_path, human_opinion)

print(human_response)





