import requests

species = [
    ('American Bison', 'Bison bison'),
    ('Green Iguana', 'Iguana iguana')
]

for name, sci in species:
    r = requests.get('https://api.inaturalist.org/v1/taxa', 
                     params={'q': sci, 'rank': 'species'}, 
                     timeout=10)
    if r.status_code == 200 and r.json().get('results'):
        result = r.json()['results'][0]
        print(f"{name}: taxon_id = {result['id']}")
