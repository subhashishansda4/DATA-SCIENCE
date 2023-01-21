import requests
from creds import KEY

cluster = 'asia'
region = 'sea'
region_ = 'sg2'

CLUSTER_URL = f"https://{cluster}.api.riotgames.com"
REGION_URL = f"https://{region}.api.riotgames.com"
REGION_URL_ = f"https://{region_}.api.riotgames.com"

class API():
    def __init__(self, token):
        self._token = token
    
    def puuid(self, game_name, tag_line):
        data = requests.get(CLUSTER_URL + f"/riot/account/v1/accounts/by-riot-id/{game_name}/{tag_line}", headers={'X-Riot-Token': self._token}).json()
        try:
            return data['puuid']
        except:
            None
    
    def result(self, puuid):
        data = requests.get(REGION_URL_ + f"/lol/summoner/v4/summoners/by-puuid/{puuid}", headers={'X-Riot-Token': self._token}).json()
        try:
            return {
                'account_id': data['accountId'],
                'summoner_id': data['id'],
                'profile_name': data['name'],
                'summoner_level': data['summonerLevel']
            }
        except:
            None
            
    def data(self, summoner_id):
        data = requests.get(REGION_URL_ + f"/lol/summoner/v4/summoners/by-puuid/{summoner_id}", headers={'X-Riot-Token': self._token}).json()
        try:
            return {
                'account_id': data['accountId'],
                'summoner_id': data['id'],
                'profile_name': data['name'],
                'summoner_level': data['summonerLevel']
            }
        except:
            None
    
    def mastery_score(self, summoner_id):
        data = requests.get(REGION_URL_ + f"/lol/champion-mastery/v4/champion-masteries/by-summoner/{summoner_id}", headers={'X-Riot-Token': self._token}).json()
        try:
            return data
        except:
            None
            
    def free_champions(self):
        data = requests.get(REGION_URL_ + "/lol/platform/v3/champion-rotations", headers={'X-Riot-Token': self._token}).json()
        return data['freeChampionIds']
        
    
    def matches(self, puuid):
        data = requests.get(REGION_URL + f"/lol/match/v5/matches/by-puuid/{puuid}/ids", headers={'X-Riot-Token': self._token}).json()
        return data
    
    
    
    
    
if __name__ == "__main__":
    api = API(KEY)
    
    game_name = 'ReynaOrThrow'
    tag_line = '3145'
    
    puuid = api.puuid(game_name, tag_line)
    result = api.result(puuid)
    
    mastery_score = api.mastery_score(result['summoner_id'])
    _ = api.free_champions()
    
    matches = api.matches(puuid)
    
    if puuid == None:
        print('NHK?')
    else:
        print(result['summoner_id'])
        print(_)
        print(mastery_score[0]['championPoints'])
        print(matches[0])
        
        
        
        