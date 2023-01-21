from flask import Flask, request
from main import API
from creds import KEY

app = Flask(__name__)

@app.route('/', methods=['POST'])
def function():
    api = API(KEY)
    game_name = request.form['game_name']
    tag_line = request.form['tag_line']
    
    puuid = api.puuid(game_name, tag_line)
    result = api.result(puuid)
    
    return '400' if puuid == None else result

if __name__ == "__main__":
    app.run(port=5000)