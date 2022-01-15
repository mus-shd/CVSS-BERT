import uvicorn

from fastapi import FastAPI

app = FastAPI()

@app.get('/')

def index():

    return {'message': "This is the home page of this API. Go to /apiv1/ or /apiv2/?name="}

@app.get('/apiv1/{name}')

def api1(name: str):

    return {'message': f'Hello! @{name}'}

@app.get('/apiv2/')

def api2(name: str):

    return {'message': f'Hello! @{name}'}

if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)