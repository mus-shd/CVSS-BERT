from flask import Flask,render_template,request
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
    

    
def CVSS_vector_and_severity_predictor(vulnerability_description: str):
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


# start flask
app = Flask(__name__)

# render default webpage
@app.route('/')
def home():
    return render_template('home.html')

# when the post method detect, then redirect to success function
@app.route('/predict', methods=['POST', 'GET'])
def get_data():
    if request.method == 'POST':
        description = request.form['description']
        predictions_dict = CVSS_vector_and_severity_predictor(description)
        cvss_vector = predictions_dict['CVSS vector']
        score = predictions_dict['severity score']
        rating = predictions_dict['severity rating'].upper()
        return render_template('home.html', vector=cvss_vector, score=score, rating=rating)
    


if(__name__=='__main__'):
    app.run(host='0.0.0.0', port=4000, debug=True, use_reloader=False)