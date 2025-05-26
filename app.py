import os
import tempfile
from pathlib import Path
import dash
from dash import dcc, html, dash_table, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import pandas as pd
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.chat_models import ChatOpenAI
from crewai import Agent, Task, Crew, Process
import plotly.express as px

# Initialize the Dash app with Bootstrap theme
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Set up the layout
app.layout = dbc.Container([
    html.H1("Financial Document Analyzer", className="text-center my-4"),
    
    # File upload section
    dbc.Card([
        dbc.CardBody([
            html.H4("Upload Financial Documents", className="card-title"),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Files')
                ]),
                style={
                    'width': '100%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px 0'
                },
                multiple=True
            ),
            html.Div(id='output-data-upload'),
        ])
    ], className="mb-4"),
    
    # Query section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Query Your Documents", className="card-title"),
                    dbc.InputGroup([
                        dbc.Input(id="query-input", placeholder="Ask a question about the documents...", type="text"),
                        dbc.Button("Search", id="search-button", color="primary")
                    ]),
                    html.Div(id="query-results", className="mt-3")
                ])
            ])
        ], width=8),
        
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Document Analysis", className="card-title"),
                    dcc.Loading(
                        id="loading-analysis",
                        type="circle",
                        children=html.Div(id="analysis-results")
                    )
                ])
            ])
        ], width=4)
    ]),
    
    # Document visualization section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Document Overview", className="card-title"),
                    dcc.Graph(id="document-graph")
                ])
            ])
        ])
    ], className="mt-4")
], fluid=True)

# Global variables
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize ChromaDB
persist_directory = os.path.join(UPLOAD_FOLDER, 'chroma_db')
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# Initialize CrewAI agents
analyst_agent = Agent(
    role='Financial Analyst',
    goal='Analyze financial documents and extract key insights',
    backstory="""You are an experienced financial analyst with expertise in analyzing 
    annual reports and other financial documents. You can identify key financial metrics, 
    trends, and potential risks.""",
    verbose=True
)

# Function to process uploaded files
def process_uploaded_files(contents, filenames):
    if not contents:
        return "No files uploaded"
    
    all_docs = []
    
    for content, filename in zip(contents, filenames):
        # Save the uploaded file
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Load the document based on file type
        if filename.lower().endswith('.pdf'):
            loader = PyPDFLoader(file_path)
        elif filename.lower().endswith(('.doc', '.docx')):
            loader = Docx2txtLoader(file_path)
        else:
            loader = TextLoader(file_path)
        
        docs = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        all_docs.extend(splits)
    
    # Create vector store
    vectordb = Chroma.from_documents(
        documents=all_docs,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    vectordb.persist()
    
    return f"Successfully processed {len(filenames)} files with {len(all_docs)} chunks."

# Callback for file upload
@callback(
    Output('output-data-upload', 'children'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filenames')
)
def update_output(contents, filenames):
    if contents is None:
        return "No files uploaded yet."
    
    try:
        result = process_uploaded_files(contents, filenames)
        return html.Div([
            html.H5("Upload Successful"),
            html.P(result),
            html.Hr(),
            html.H5("Uploaded Files:"),
            html.Ul([html.Li(filename) for filename in filenames])
        ])
    except Exception as e:
        return html.Div([
            html.Div('An error occurred while processing the files.'),
            html.Div(str(e))
        ])

# Callback for document query
@callback(
    Output('query-results', 'children'),
    Input('search-button', 'n_clicks'),
    State('query-input', 'value')
)
def query_documents(n_clicks, query):
    if n_clicks is None or not query:
        return ""
    
    try:
        # Load the vector store
        vectordb = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        
        # Create retriever
        retriever = vectordb.as_retriever(search_kwargs={"k": 3})
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        
        # Get response
        result = qa_chain({"query": query})
        
        # Format the response
        response = html.Div([
            html.H5("Answer:"),
            html.P(result["result"]),
            html.Hr(),
            html.H5("Sources:"),
            html.Ul([
                html.Li(f"Page {doc.metadata.get('page', 'N/A')}: {doc.page_content[:200]}...") 
                for doc in result["source_documents"]
            ])
        ])
        
        return response
    except Exception as e:
        return html.Div([
            html.Div('An error occurred while querying the documents.'),
            html.Div(str(e))
        ])

# Callback for document analysis
@callback(
    Output('analysis-results', 'children'),
    Input('search-button', 'n_clicks'),
    State('query-input', 'value')
)
def analyze_document(n_clicks, query):
    if n_clicks is None or not query:
        return "Submit a query to analyze the documents."
    
    try:
        # Create analysis task
        analysis_task = Task(
            description=f"Analyze the financial documents and provide insights about: {query}",
            agent=analyst_agent,
            expected_output="A detailed analysis with key insights and recommendations."
        )
        
        # Create and run crew
        crew = Crew(
            agents=[analyst_agent],
            tasks=[analysis_task],
            verbose=2
        )
        
        # Get analysis results
        result = crew.kickoff()
        
        return html.Div([
            html.H5("Analysis Results:"),
            html.P(result)
        ])
    except Exception as e:
        return html.Div([
            html.Div('An error occurred during document analysis.'),
            html.Div(str(e))
        ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
