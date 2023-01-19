import pickle
from flask import Flask, request
from pred import PROCESS, feature

app = Flask(__name__)

with open("A:/O/projects/DATA SCIENCE/Sentence-Similarity/pickle/grid.pkl", "rb") as file:
    similarity_model = pickle.load(file)

@app.route('/', methods=['POST'])
def predict():
    sentence1 = request.form['sentence1']
    sentence2 = request.form['sentence2']
    process = PROCESS()
    features = feature(sentence1, sentence2, process)
    try:
        similarity = similarity_model.predict_proba(features)[:, 1]
        return 'similar' if similarity>0.75 else 'unique'
    except:
        return 'unique'
    
if __name__ == '__main__':
    app.run(port=5000)

