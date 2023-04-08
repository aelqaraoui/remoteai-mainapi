import os
import pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('distilbert-base-nli-mean-tokens')

from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv('.env')

# Connect to the MongoDB database
client = MongoClient(os.environ.get("MONGODB_URI"))
db = client['JOBS']
collection = db['listings']

pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"), environment='us-east4-gcp')
print(pinecone.list_indexes())

index_name = pinecone.list_indexes()[0]

index = pinecone.Index(index_name=index_name)

def query(test_article, namespace, field, top_k=30):
    '''Queries an article using its title in the specified
     namespace and prints results.'''

    # Create vector embeddings based on the title column
    encoded_titles = model.encode(test_article[field], 
                                  show_progress_bar=False)
    test_article[field + '_vector'] = encoded_titles.tolist()

    # Query namespace passed as parameter using title vector
    query_result_titles = index.query(test_article[field + '_vector'], 
                                      namespace=namespace, 
                                      top_k=top_k)

    return query_result_titles


from fastapi import FastAPI
from typing import Dict
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # List of allowed origins (use "*" to allow all origins)
    allow_credentials=True,
    allow_methods=["*"],  # List of allowed methods (use "*" to allow all methods)
    allow_headers=["*"],  # List of allowed headers (use "*" to allow all headers)
)

@app.post("/process_data")
async def process_data(job: Dict):

    fields = list(job.keys())

    results = []
    for field in fields:
        for namespace in ['title', 'jobType', 'jobLevel', 'location', 'salary', 'company', 'description']:
            matches = query(job, namespace, field)['matches']
            for match in matches:
                if namespace == 'title':
                    results.append({
                        'id': match['id'],
                        'score': 3 * match['score']
                    })
                elif namespace == 'description':
                    results.append({
                        'id': match['id'],
                        'score': 2 * match['score']
                    })
                else:
                    results.append(match)

    counts = {}
    scores = {}

    for res in results:

        if not (res['id'] in counts.keys()):
            counts[res['id']] = 1
            scores[res['id']] = res['score']

        else:
            counts[res['id']] += 1
            scores[res['id']] += res['score']

    job_matches = sorted(scores.items(), key=lambda x:x[1], reverse=True)[:30]

    data = []
    for job_match in job_matches:

        print(job_match)

        pl = collection.find_one({'hash': job_match[0]})

        print("PL", pl)

        del pl['_id']
        data.append(pl)

    response = {
        'status': 'success',
        'message': 'JSON data received and processed',
        'matches': data
    }

    return response

@app.get("/")
def home():
    return {"message":"Hello MAIINAPI"}