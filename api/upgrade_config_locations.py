import json
import os

CONFIG_PATH = r"C:\Users\nars7\wildlife-id\api\species_config.json"

# Manual "Hotspot" Database (AI Knowledge Injection)
# Maps Species Name -> List of Top Viewing Locations
SPECIES_LOCATIONS = {
    # --- MAMMALS ---
    "American Bison": ["Yellowstone National Park, WY", "Custer State Park, SD", "Antelope Island State Park, UT"],
    "American Black Bear": ["Great Smoky Mountains National Park, TN", "Shenandoah National Park, VA", "Yosemite National Park, CA"],
    "Grizzly Bear": ["Yellowstone National Park, WY", "Glacier National Park, MT", "Denali National Park, AK"],
    "Moose": ["Grand Teton National Park, WY", "Isle Royale National Park, MI", "Baxter State Park, ME"],
    "Elk": ["Rocky Mountain National Park, CO", "Yellowstone National Park, WY", "Olympic National Park, WA"],
    "Mountain Goat": ["Glacier National Park, MT", "Mount Rainier National Park, WA", "Olympic National Park, WA"],
    "Bighorn Sheep": ["Rocky Mountain National Park, CO", "Badlands National Park, SD", "Zion National Park, UT"],
    "Gray Wolf": ["Yellowstone National Park, WY", "Isle Royale National Park, MI", "Voyageurs National Park, MN"],
    "Cougar": ["Big Bend National Park, TX", "Olympic National Park, WA", "Everglades National Park, FL (Florida Panther)"],
    "Bobcat": ["Point Reyes National Seashore, CA", "Big Cypress National Preserve, FL", "Kiawah Island, SC"],
    "Coyote": ["Joshua Tree National Park, CA", "Yellowstone National Park, WY", "Death Valley National Park, CA"],
    "Red Fox": ["San Juan Island National Historical Park, WA", "Acadia National Park, ME", "Prince Edward Island National Park, Canada"],
    "Arctic Fox": ["Arctic National Wildlife Refuge, AK", "Pribilof Islands, AK", "Churchill, Manitoba"],
    "Polar Bear": ["Churchill, Manitoba (Canada)", "Kaktovik, AK", "Svalbard (Norway)"],
    "Sea Otter": ["Monterey Bay, CA", "Kenai Fjords National Park, AK", "Olympic Coast, WA"],
    "American Beaver": ["Voyageurs National Park, MN", "Acadia National Park, ME", "Rocky Mountain National Park, CO"],
    "Raccoon": ["Golden Gate Park, CA", "Central Park, NY", "Everglades National Park, FL"],
    "Virginia Opossum": ["Suburban Areas (Eastern US)", "Great Smoky Mountains, TN", "Ozark National Forest, AR"],
    "Nine-Banded Armadillo": ["Cumberland Island National Seashore, GA", "Big Thicket National Preserve, TX", "Everglades National Park, FL"],
    "Striped Skunk": ["Suburban Areas (North America)", "Shenandoah National Park, VA", "Point Reyes, CA"],
    
    # --- BIRDS ---
    "Bald Eagle": ["Haines, AK (Chilkat Bald Eagle Preserve)", "Upper Mississippi River, IA/IL", "Klamath Basin, OR/CA"],
    "Golden Eagle": ["Denali National Park, AK", "Snake River Birds of Prey, ID", "Diablo Range, CA"],
    "Peregrine Falcon": ["Acadia National Park, ME", "Zion National Park, UT", "New York City Skyscrapers, NY"],
    "Great Horned Owl": ["Forest Park, MO", "Sax-Zim Bog, MN", "Everywhere in North America (Nocturnal)"],
    "Snowy Owl": ["Logan Airport, MA", "Duluth Harbor, MN", "Jones Beach State Park, NY"],
    "Wild Turkey": ["Great Smoky Mountains, TN", "Land Between the Lakes, KY/TN", "Suburban Northeast US"],
    "California Condor": ["Grand Canyon National Park, AZ", "Pinnacles National Park, CA", "Bitter Creek National Wildlife Refuge, CA"],
    "Sandhill Crane": ["Platte River, NE (Migration)", "Bosque del Apache, NM", "Okeefenokee Swamp, GA"],
    "Whooping Crane": ["Aransas National Wildlife Refuge, TX", "Wood Buffalo National Park (Canada)", "Necedah National Wildlife Refuge, WI"],
    "Trumpeter Swan": ["Yellowstone National Park, WY", "Create Meadows National Wildlife Refuge, ID", "Red Rock Lakes, MT"],
    
    # --- REPTILES/AMPHIBIANS ---
    "American Alligator": ["Everglades National Park, FL", "Okefenokee Swamp, GA", "Brazos Bend State Park, TX"],
    "American Crocodile": ["Everglades National Park (Flamingo), FL", "Crocodile Lake National Wildlife Refuge, FL", "Biscayne National Park, FL"],
    "Gila Monster": ["Saguaro National Park, AZ", "Sonoran Desert National Monument, AZ", "Tonto National Monument, AZ"],
    "Green Iguana": ["Key West, FL", "Miami, FL (Suburban Areas)", "Puerto Rico"],
    "Komodo Dragon": ["Komodo National Park (Indonesia)", "National Zoo, Washington DC (Captive)", "San Diego Zoo (Captive)"], # Exotic
    
    # --- INVASIVE/INTRODUCED ---
    "Wild Boar": ["Great Smoky Mountains, TN", "Big South Fork, TN/KY", "Congaree National Park, SC"],
    "Nutria": ["Blackwater National Wildlife Refuge, MD", "Jean Lafitte National Historical Park, LA", "Willamette Valley, OR"],
    "Burmese Python": ["Everglades National Park, FL", "Big Cypress National Preserve, FL", "Southern Florida Waterways"],
}

def inject_locations():
    """Reads the JSON config, adds 'hotspots' from the dictionary, and saves back."""
    if not os.path.exists(CONFIG_PATH):
        print(f"❌ Config not found at {CONFIG_PATH}")
        return

    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        updated_count = 0
        for entry in data:
            name = entry["name"]
            # Inject hotspots if we have knowledge about it
            if name in SPECIES_LOCATIONS:
                entry["hotspots"] = SPECIES_LOCATIONS[name]
                updated_count += 1
            else:
                # Default Logic: If no specific data, maybe suggest "National Parks in [Origin]"?
                # For now, leave empty or generic.
                if "hotspots" not in entry:
                     entry["hotspots"] = [f"Natural Reserves in {entry['origin']}"]

        with open(CONFIG_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        print(f"✅ Successfully injected locations for {updated_count} species.")
        print(f"📁 Config saved to {CONFIG_PATH}")

    except Exception as e:
        print(f"❌ Error updating config: {e}")

if __name__ == "__main__":
    inject_locations()
