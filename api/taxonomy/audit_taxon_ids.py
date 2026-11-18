"""
Audit and fix taxon_ids in species_config.json
Checks each species against iNaturalist API to verify taxon_ids are correct.
"""

import json
import requests
import time

def get_correct_taxon_id(scientific_name: str) -> dict:
    """Look up the correct taxon_id from iNaturalist by scientific name"""
    url = f"https://api.inaturalist.org/v1/taxa"
    params = {"q": scientific_name, "rank": "species,subspecies", "per_page": 5}
    
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            results = response.json().get("results", [])
            for r in results:
                if r.get("name", "").lower() == scientific_name.lower():
                    return {
                        "taxon_id": r["id"],
                        "name": r["name"],
                        "common_name": r.get("preferred_common_name", ""),
                        "rank": r.get("rank", "")
                    }
            # Return first result if no exact match
            if results:
                return {
                    "taxon_id": results[0]["id"],
                    "name": results[0]["name"],
                    "common_name": results[0].get("preferred_common_name", ""),
                    "rank": results[0].get("rank", "")
                }
    except Exception as e:
        print(f"  Error: {e}")
    return None

def main():
    with open("species_config.json", "r") as f:
        species_list = json.load(f)
    
    print("🔍 Auditing taxon_ids...\n")
    
    issues = []
    for i, species in enumerate(species_list):
        name = species["name"]
        scientific = species.get("scientific_name", "")
        current_id = species.get("taxonomy", {}).get("taxon_id")
        
        print(f"[{i+1}/{len(species_list)}] {name} ({scientific})")
        print(f"  Current taxon_id: {current_id}")
        
        # Look up correct ID
        correct = get_correct_taxon_id(scientific)
        if correct:
            print(f"  iNaturalist: {correct['taxon_id']} - {correct['common_name']} ({correct['name']})")
            
            if correct["taxon_id"] != current_id:
                issues.append({
                    "name": name,
                    "scientific_name": scientific,
                    "current_id": current_id,
                    "correct_id": correct["taxon_id"],
                    "inat_name": correct["common_name"]
                })
                print(f"  ⚠️  MISMATCH!")
            else:
                print(f"  ✓ OK")
        else:
            print(f"  ❌ Not found on iNaturalist")
            issues.append({
                "name": name,
                "scientific_name": scientific,
                "current_id": current_id,
                "correct_id": None,
                "inat_name": "NOT FOUND"
            })
        
        time.sleep(0.5)  # Rate limiting
        print()
    
    print("\n" + "="*50)
    print(f"SUMMARY: {len(issues)} issues found\n")
    
    for issue in issues:
        print(f"  {issue['name']}: {issue['current_id']} -> {issue['correct_id']} ({issue['inat_name']})")
    
    # Save issues to file
    with open("taxon_id_issues.json", "w") as f:
        json.dump(issues, f, indent=2)
    print(f"\nSaved issues to taxon_id_issues.json")

if __name__ == "__main__":
    main()
