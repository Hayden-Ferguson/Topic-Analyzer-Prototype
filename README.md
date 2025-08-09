<h1>Topic Analyzer Prototype</h1>

<h2>Introduction</h2>
This project aims to use AI to analyze articles and find general trends in specified topics and provide sources for each trend. Using Google's Gemini and Pinecone's databases, 
this application can scan text files and summarize some trends found in relevant topics. This makes it a good aid in researching the topics you provide it.
Due to being a prototype, a method of collecting sources is not provided.

<h2>Requirements</h2>
Python 3.11 or higher
Pydantic 2.0 or higher
Pinecone API Key
Gemini API Key

<h2>Installation</h2>

1. Install the requiremed dependencies:

```pip install -r requirements.txt```

2. Set up your Gemini API key:

```export GEMINI_API_KEY='your-api-key-here'  # On Windows, for coomand prompt use: set GEMINI_API_KEY=your-api-key-here #For Powershell use $env:GEMINI_API_KEY="your_api_key_here"```

3. Set up your Pinecone API key:

```export PINECONE_API_KEY='your-api-key-here'  # On Windows, for coomand prompt use: set PINECONE_API_KEY=your-api-key-here #For Powershell use $env:PINECONE_API_KEY="your_api_key_here"```

<h2>Usage</h2>
Add any sources you want to use into the /sources folder as .txt files.

Whenever you want update your database of sources, such as the first time you run this, add "read" to your command line, like so:

```python .\main.py read```

Otherwise, to save time and resources, run it without adding "read":

```python .\main.py```

<h2>License</h2>
This project is provided "as is" under the MIT License. Feel free to use, modify, and distribute it as you wish.
