from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
import torch
from transformers import pipeline
from transformers import BertTokenizerFast
from transformers import BertForSequenceClassification
import torch.nn.functional as F
import pickle
from cvss import CVSS3


print("loading tokenizer and models...")

tokenizer = BertTokenizerFast.from_pretrained('prajjwal1/bert-small')

print("tokenizer loaded")

CVSS_classifiers_dict = {}
for metric in ['attack_vector', 'attack_complexity', 'privileges_required', 'user_interaction',
'scope', 'confidentiality_impact', 'integrity_impact', 'availability_impact']:
    model = BertForSequenceClassification.from_pretrained('./models/bert-small-vulnerability_'+metric+'-classification')
    model.eval()
    with open('labels/'+metric+'_label.txt', 'rb') as f:
        labels = pickle.load(f)
    CVSS_classifiers_dict[metric] = {'model': model, 'labels': labels}
    print(metric+" model loaded")


class Vulnerability(BaseModel):
    description: str
    
app = FastAPI()

@app.get('/')
def index():

    return {'message': "This is the home page of this API."}



@app.post('/predict')
def CVSS_vector_and_severity_predictor(vulnerability: Vulnerability):
    vulnerability_description = vulnerability.description
    tokenized_description = tokenizer(vulnerability_description, truncation=True, padding=True, max_length=128)
    input_ids = torch.tensor(tokenized_description['input_ids']).unsqueeze(0)
    attention_mask = torch.tensor(tokenized_description['attention_mask']).unsqueeze(0)
    
    predictions_dict = {}
    cvss_vector = "CVSS:3.1"
        
    for metric in ['attack_vector', 'attack_complexity', 'privileges_required', 'user_interaction',
'scope', 'confidentiality_impact', 'integrity_impact', 'availability_impact']:
        outputs = CVSS_classifiers_dict[metric]['model'](input_ids, attention_mask=attention_mask)
        predicted_class = torch.max(F.softmax(outputs.logits, dim=1), dim=1)[1].tolist()[0]
        predictions_dict[metric] = CVSS_classifiers_dict[metric]['labels'][predicted_class]
    
    cvss_vector = "CVSS:3.1/AV:"+predictions_dict['attack_vector'][0]+"/AC:"+predictions_dict['attack_complexity'][0]+"/PR:" \
    +predictions_dict['privileges_required'][0]+"/UI:"+predictions_dict['user_interaction'][0]+"/S:"+predictions_dict['scope'][0] \
    +"/C:"+predictions_dict['confidentiality_impact'][0]+"/I:"+predictions_dict['integrity_impact'][0]+"/A:"+predictions_dict['availability_impact'][0]
    
    predictions_dict['CVSS vector'] = cvss_vector
    
    c = CVSS3(cvss_vector)
    predictions_dict['severity score'] = c.scores()[0]
    predictions_dict['severity rating'] = c.severities()[0]
        
    return predictions_dict



if __name__ == '__main__':

    uvicorn.run(app, host='0.0.0.0', port=8000, debug=True)