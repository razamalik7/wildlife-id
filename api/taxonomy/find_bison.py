import requests

# Find correct American Bison ID
r = requests.get('https://api.inaturalist.org/v1/taxa', 
                 params={'q': 'Bison bison', 'rank': 'species'})
results = r.json()['results']

print("American Bison search results:")
for x in results[:5]:
    print(f"  {x['id']}: {x['name']} ({x.get('preferred_common_name','')}) - {x.get('iconic_taxon_name','')}")
