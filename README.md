

## Introduction

Introduction to LangGraph! 

## Setup

### Python version

Make sure you're using Python version 3.11, 3.12, or 3.13.
```
python3 --version
```

### Create an environment and install dependencies
#### Mac/Linux/WSL
```
$ python3 -m venv venv
$ source venv/bin/activate
$ pip install -r requirements.txt
```

### Running notebooks
If you don't have Jupyter set up, follow the installation instructions [here](https://jupyter.org/install).
```
$ jupyter notebook
```

### Setting up env variables
Briefly going over how to set up environment variables. 
#### Mac/Linux/WSL
```
$ export API_ENV_VAR="your-api-key-here"
```
#### Windows Powershell
```
PS> $env:API_ENV_VAR = "your-api-key-here"
```

### Set OpenAI API key
* If you don't have an OpenAI API key, you can sign up [here](https://openai.com/index/openai-api/).
*  Set `OPENAI_API_KEY` in your environment 

### Sign up and Set LangSmith API
* Sign up for LangSmith [here](https://docs.langchain.com/langsmith/create-account-api-key#create-an-account-and-api-key), find out more about LangSmith and how to use it within your workflow [here](https://www.langchain.com/langsmith). 
*  Set `LANGSMITH_API_KEY`, `LANGSMITH_TRACING_V2="true"` `LANGSMITH_PROJECT="langchain-academy"`in your environment 
*  If you are on the EU instance also set `LANGSMITH_ENDPOINT`="https://eu.api.smith.langchain.com" as well.

### Set up Tavily API for web search

* Tavily Search API is a search engine optimized for LLMs and RAG, aimed at efficient, 
quick, and persistent search results. 
* You can sign up for an API key [here](https://tavily.com/). 
It's easy to sign up and offers a very generous free tier. Some lessons (in Module 4) will use Tavily. 

* Set `TAVILY_API_KEY` in your environment.

### Set up Studio

* Studio is a custom IDE for viewing and testing agents.
* Studio can be run locally and opened in your browser on Mac, Windows, and Linux.
* See documentation [here](https://docs.langchain.com/langsmith/studio#local-development-server) on the local Studio development server. 
* Graphs for LangGraph Studio are in the `module-x/studio/` folders for module 1-5.
* To start the local development server, make sure your virtual environment is active and run the following command in your terminal in the `/studio` directory in each module:

```
langgraph dev
```

You should see the following output:
```
- 🚀 API: http://127.0.0.1:2024
- 🎨 Studio UI: https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- 📚 API Docs: http://127.0.0.1:2024/docs
```

Open your browser and navigate to the Studio UI: `https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`.

* To use Studio, you will need to create a .env file with the relevant API keys
* Run this from the command line to create these files for module 1 to 5, as an example:
```
for i in {1..5}; do
  cp module-$i/studio/.env.example module-$i/studio/.env
  echo "OPENAI_API_KEY=\"$OPENAI_API_KEY\"" > module-$i/studio/.env
done
echo "TAVILY_API_KEY=\"$TAVILY_API_KEY\"" >> module-4/studio/.env
```
