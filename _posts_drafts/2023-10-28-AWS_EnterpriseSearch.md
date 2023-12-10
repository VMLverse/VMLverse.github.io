---
title:  "LLM Based Enterprise Search Solution using RAG In AWS"
category: posts
date: 2023-10-28
excerpt: "Building Large Language Models for Complex Enterprise Search Tasks using Retrieval Augmented Generation (RAG) - In AWS"
header:
  teaser: /assets/images/2023-10-28-AWS_EnterpriseSearch/IMG_1868t.jpeg
toc: false
toc_label: "Contents"
tags:
  - AWS
  - LLM
  - enterprise search
  - RAG
---

## Introduction
- Within the field of Generative AI, Text generation, especially Enterprise Search for ChatBots, Agent Assits, Internal Forums is a very common use case.
- Training LLMs on Enterprise data will not be effective as it is very complex and knowledge-intensive task. 
- Retrieval Augmented Generation (RAG) is concept introduced by Meta AI researchers to combine an information retrieval component with a text generator model. RAG can be fine-tuned and its internal knowledge can be modified in an efficient manner and without needing retraining of the entire model.

 
## Overview of Solution
- Solution uses transformer model for answering questions
- Zero-shot prompting for generating answers from untrained data
- Benefits of the solution:
  - Accurate answers from internal documents
  - Time-saving with Large Language Models (LLMs)
  - Centralized dashboard for previous questions
  - Stress reduction from manual information search

### Retrivel Augmented Generation (RAG)
- Retrieval Augmented Generation (RAG) enhances LLM-based queries
- Addresses shortcomings of LLM-based queries
- RAG can be implemented using Amazon Kendra
- Risks and limitations of LLM-based queries without RAG:
  - Hallucinations (probability based answering) leads to inaccurate answers
  - Multiple data source challenges
  - Security and privacy concerns: possibility of unintentionally surfacing personal or sensitive information
  - Data relevance: information is often not current
  - Cost considerations for deployment: Running LLMs can require substantial computational resources
![jpg](/assets/images/2023-10-28-AWS_EnterpriseSearch/IMG_1867t.jpeg)

### Why does RAG work?
- An application using the RAG approach retrieves information most relevant to the user’s request from the enterprise knowledge base or content, bundles it as context along with the user’s request as a prompt, and then sends it to the LLM to get a GenAI response.
- LLMs have limitations around the maximum word count for the input prompt, therefore choosing the right passages among thousands or millions of documents in the enterprise, has a direct impact on the LLM’s accuracy.

![jpg](/assets/images/2023-10-28-AWS_EnterpriseSearch/IMG_1868t.jpeg)

### Why Amazon Kendra is required?
- We have already established Effective RAG (Retrieval Augmented Generation) is crucial for accurate AI responses.
- Content retrieval is a key step for providing context to the AI model.
- Amazon Kendra index is used to ingest enterprise unstructured data from data sources such as wiki pages, MS SharePoint sites, Atlassian Confluence, and document repositories such as Amazon S3. 
- Amazon Kendra's intelligent search plays a vital role in this process.
- Kendra offers semantic search, making it easy to find relevant content.
- Amazon Kendra can also act as web-crawler - You specify a URL and specify how deep it needs to go in the path name and it will retrieve all that data for you. 
- It doesn't require advanced machine learning knowledge.
- Kendra provides a Retrieve API for obtaining relevant passages.
- It supports various data sources and formats, with access control.
- It integrates with user identity providers for permissions control.

### Solution Workflow

1. User sends a request to the GenAI app.
2. App queries Amazon Kendra index based on the request.
3. Kendra provides search results with excerpts from enterprise data.
4. App sends user request and retrieved data to LLM.
5. LLM generates a concise response.
6. Response is sent back to the user.

![png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/05/02/ML-13807-image001-new.png)

### Choice of LLM
- Architecture allows choosing the right LLM for the use case.
- LLM options include Amazon partners (Hugging Face, AI21 Labs, Cohere) and others hosted on Amazon SageMaker endpoints.
- With Amazon Bedrock, choose Amazon Titan, partner LLMs (e.g., AI21 Labs, Anthropic) securely within AWS.
- Benefits of Amazon Bedrock: serverless architecture, single API for LLMs, and streamlined developer workflow.

## GenAI App
- For the best results, a GenAI app needs to engineer the prompt based on the user request and the specific LLM being used. 
- Conversational AI apps also need to manage the chat history and the context. 
- Both these tasks can be accomplished using LangChain

## What is LangChain?
- [LangChain](https://python.langchain.com/docs/get_started/introduction) is an open-source framework for developing applications powered by language models. It enables applications that are context Aware and Reason.
- Developers can utilize LangChain frameworks for LLM integration and orchestration.
- AWS's [AmazonKendraRetriever](https://python.langchain.com/docs/integrations/retrievers/amazon_kendra_retriever) class implements a LangChain retriever interface, which applications can use in conjunction with other LangChain interfaces such as [chains](https://python.langchain.com/docs/modules/chains.html) to retrieve data from an Amazon Kendra index. - AmazonKendraRetriever class uses Amazon Kendra’s Retrieve API to make queries to the Amazon Kendra index and obtain the results with excerpt passages that are most relevant to the query.

## A Complete Sample AWS Workflow:
AWS Workflow of Question Answering over Documents:
  1. User query via web interface
  2. Authentication with Amazon Cognito
  3. Front-end hosted on AWS Amplify
  4. Amazon API Gateway with authenticated REST API
  5. PII redaction with Amazon Comprehend:
     - User query analyzed for PII
     - Extract PII entities
  6. Information retrieval with Amazon Kendra:
     - Index of documents for answers
     - LangChain QA retrieval for user queries
  7. Integration with Amazon SageMaker JumpStart:
     - AWS Lambda with LangChain library
     - Connect to SageMaker for inference
  8. Store responses in Amazon DynamoDB with user query and metadata
  9. Return response via Amazon API Gateway REST API.
![png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/09/13/DBSBLOG-14696_Image_1-1.png)

### AWS Lambda Functions Flow:
  1. Check and redact PII/Sensitive info
  2. LangChain QA Retrieval Chain
     - Search and retrieve relevant info
  3. Context Stuffing & Prompt Engineering with LangChain
  4. Inference with LLM (Large Language Model)
  5. Return response & Save it

![png](https://d2908q01vomqb2.cloudfront.net/f1f836cb4ea6efb2a0b1b99f41ad8b103eff4b59/2023/09/12/DBSBLOG-14696_Image_2-1.png)


### Security in this Workflow
- Security best practices documented in [Well-Architected Framework](https://aws.amazon.com/architecture/well-architected/?wa-lens-whitepapers.sort-by=item.additionalFields.sortDate&wa-lens-whitepapers.sort-order=desc&wa-guidance-whitepapers.sort-by=item.additionalFields.sortDate&wa-guidance-whitepapers.sort-order=desc)
- [Amazon Cognito](https://docs.aws.amazon.com/cognitoidentity/latest/APIReference/Welcome.html) for authentication
- Integration with third-party identity providers are available
- Traceability through user identification
- [Amazon Comprehend](https://docs.aws.amazon.com/comprehend/latest/dg/realtime-pii-api.html) for PII detection and redaction
- Redacted PII includes sensitive data
- User-provided PII is not stored or used by Amazon Kendra or LLM

### What is the difference between Amazon SageMaker & Amazon Bedrock?
**Amazon Bedrock:**  
- Amazon Bedrock provides access to a range of pre-trained foundation models (text, image etc).
- Bedrock is serverless, meaning there are no servers or infrastructure to manage. Users only need to interact with a simple API
- Easy model customization through fine-tuning. Customers only need to point Bedrock to a few labeled examples in an S3 bucket.
- None of the user data is used to train the underlying foundational models
- Bedrock integrates seamlessly with other AWS services
**Amazon SageMaker**
- comprehensive service that allows us to build, train, and deploy machine learning models for extended use cases.
- supports every step of the machine learning process, from data labeling and preparation to model deployment and monitoring
- provides various built-in algorithms and frameworks
- automatically tunes models
- provides a single, web-based visual interface where you can perform all ML development steps, making it easier to build, train, and tune machine learning models.
- Managed Spot Training allows users to use Amazon EC2 Spot instances for training ML models, resulting in cost savings of up to 90% compared to on-demand instances

## Building the RAG based retrieval app based on this link


Github sample code: https://github.com/aws-samples/amazon-kendra-langchain-extensions 

Main Article: https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/ 

## Which llama2 did I choose and why?


## Deploy Llama2 on Sagemaker

https://www.philschmid.de/sagemaker-llama-llm 

## Which option to choose - Sagemaker vs Bedrock?

Llama 2 on sagemaker - a benchmark: https://huggingface.co/blog/llama-sagemaker-benchmark 

## Issue: SagemakerEndpoint model doesn't return full output..only when prompted with langchain 



## Issue: Running into Token Limit errors:

In LangChain, there are several approaches to address this concern:

Text Splitting: LangChain offers classes like SentenceTransformersTokenTextSplitter and RecursiveCharacterTextSplitter (along with its subclasses) to break down text into segments that stay within the model's maximum context length. You can find detailed information about these classes in the text_splitter.py file.

Token Counting: LangChain utilizes the tiktoken Python package to tally the number of tokens in documents, ensuring they remain below a specified limit. The get_num_tokens_from_messages method calculates the token count for a list of messages. More details about this method can be found in the openai.py file.

Limiting Results: In certain scenarios, you can restrict the number of results returned from the store based on token limits by setting reduce_k_below_max_tokens=True. This solution appears effective primarily with chat models like GPT-3.5/GPT-4. Another option is to set max_tokens_limit=4097 in ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), vectorstore.as_retriever(search_kwargs={"k": 1}), max_tokens_limit=4097).

It's important to note that these are general solutions and may require adjustments based on your specific use case. If challenges persist even after implementing these solutions, please share more details about your use case and the specific code you're using so that we can offer more targeted assistance.

https://github.com/langchain-ai/langchain/issues/12264 

## Running Stremlit App on Sagemaker

### Install dependencies
```pip install --no-cache-dir -r requirements.txt```

-The inclusion of the --no-cache-dir flag serves as a cache disabler.
- The cache, a storage mechanism, plays a crucial role in retaining installation files (.whl) for modules installed via pip. 
- It also preserves source files (.tar.gz) to prevent unnecessary re-downloading, provided they haven't expired.
- Utilizing this flag proves beneficial when storage space is limited on our hard drive or when aiming to keep Docker images as compact as possible. By employing this flag, the command can smoothly execute, optimizing memory usage throughout the process.

Next we install

```
sudo yum install -y iproute
sudo yum install -y jq
sudo yum install -y lsof
```

- **iproute:** Configures and manages network interfaces, routing tables, and tunnels using the versatile `ip` command.
  
- **lsof:** Lists open files and provides details about files opened by processes, including network connections, files, and directories.

- **jq:** Lightweight command-line JSON processor for parsing, manipulating, and extracting specific fields from JSON data.

### Run Streamlit and Create Shareable Link

The command has been bundled up into the run.sh file.

Here's a summary of what the run.sh file does:
- The script captures the current date and defines color codes for better output formatting.
- It takes an S3 path as a parameter.
- The Streamlit app (`app.py`) is executed, and its output is saved to a temporary text file (`temp.txt`).
- The script extracts the last four digits of the port number from the Network URL in the temporary text file.
- It retrieves Studio domain information, including Domain ID, Resource Name, and Resource ARN, from metadata files.
- The script extracts relevant details from the obtained information, such as the Studio domain region.
- It checks if the Studio setup is a Collaborative Space and configures the Studio URL accordingly.
- If it's a Collaborative Space, the script prompts for and validates the Space ID, saving it for future use.
- The final output includes the Studio URL and a direct link to access the Streamlit app within the Studio environment.


During development, it can be advantageous to automatically rerun the script whenever there is a modification to app.py on disk. To achieve this, we can customize the runOnSave configuration option by appending the --server.runOnSave true flag to our command:

```streamlit run app.py --server.runOnSave true```

### Check status

sh status.sh

https://aws.amazon.com/blogs/machine-learning/build-streamlit-apps-in-amazon-sagemaker-studio/

## References:
- Lewis, P. S. H., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. CoRR, abs/2005.11401. https://arxiv.org/abs/2005.11401
- LinkedIn: Transforming Question Answering with OpenAI and LangChain: Harnessing the Potential of Retrieval Augmented Generation (RAG) [link](https://www.linkedin.com/pulse/transforming-question-answering-openai-langchain-harnessing-routu/)
- AWS: Simplify access to internal information using Retrieval Augmented Generation and LangChain Agents
[link](https://aws.amazon.com/blogs/machine-learning/simplify-access-to-internal-information-using-retrieval-augmented-generation-and-langchain-agents/)
- AWS: Quickly build high-accuracy Generative AI applications on enterprise data using Amazon Kendra, LangChain, and large language models (Step-By-Step Tutorial) [link](https://aws.amazon.com/blogs/machine-learning/quickly-build-high-accuracy-generative-ai-applications-on-enterprise-data-using-amazon-kendra-langchain-and-large-language-models/)
- LangChain [link](https://python.langchain.com/docs/get_started/introduction)
- AWS Kendra Langchain Extensions - Github [link](https://github.com/aws-samples/amazon-kendra-langchain-extensions)
- AWS Kendra Demo YouTube [link](https://www.youtube.com/watch?v=NJoEyIZ_Tas&ab_channel=AWSDevelopers)
- Machine Learning How to Start in AWS [link](https://aws.amazon.com/getting-started/decision-guides/machine-learning-on-aws-how-to-choose/)
- AWS ML Workshop With UseCases [link](https://catalog.us-east-1.prod.workshops.aws/workshops/a4bdb007-5600-4368-81c5-ff5b4154f518/en-US)