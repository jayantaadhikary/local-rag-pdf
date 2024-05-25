## Local RAG query tool for PDFs

This is a simple Retrieval Augmented Generation (RAG) tool built in Python which allows us to read information from a PDF document and then generate a response based on the information in the document.

We use Ollama to run the tool Locally and we use llama3 currently as the model for the RAG tool.

### Setup

After cloning the repository, you can install the required packages using the requirements.txt file.
To install the packages, you can run the following command:

```bash
pip install -r requirements.txt
```

After installing the required packages, you also need to place the PDF documents you want to get information about in the `data` folder.

You also need to download [Ollama](https://ollama.com) if you haven't already.
After downloading Ollama, make sure you download the 'llama3' model in the terminal by running the following command:

```bash
ollama pull llama3
```

This will download the llama3 model which we will use for the RAG tool.
Learn more about Ollama implementation in my [Guide to Ollama](https://jayadky.notion.site/Guide-to-install-LLMs-locally-using-Ollama-c1a2745ed1224a3e9970c6cba5576089).
You can also download other models if you want but then you will have to change the `MODEL_NAME` in the `main.py` file.

### Usage

To use the tool, you can run the following command:

```bash
python main.py
```

This will start the script and you can then input a question and it will generate a response based on the information in the PDF document.
