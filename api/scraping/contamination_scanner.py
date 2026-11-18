"""
Quick contamination scanner - just identifies which species need re-scraping
If you see ANY bad image, mark species as contaminated and move on
"""
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random
import json

DATA_DIR = 'training_data_cropped'

# Track results
current_decision = None

def on_key(event):
    """Handle keyboard events"""
    global current_decision
    if event.key in ['c', 'enter']:
        current_decision = 'clean'
    elif event.key == 'x':
        current_decision = 'contaminated'
    elif event.key == 'n':
        current_decision = 'next'
    elif event.key == 'u':  # Uncertain - flag for expert review
        current_decision = 'uncertain'
    elif event.key == 'q':
        current_decision = 'quit_all'
    plt.close()


def scan_species(species_name, sample_size=8):
    """Scan a species to check for contamination"""
    global current_decision
    
    samples_per_split = {'train': 6, 'val': 2}
    all_images = []
    
    for split, num in samples_per_split.items():
        species_dir = os.path.join(DATA_DIR, split, species_name)
        if not os.path.exists(species_dir):
            continue
        
        images = [f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        sampled = random.sample(images, min(num, len(images)))
        all_images.extend([(os.path.join(species_dir, img), split) for img in sampled])
    
    if not all_images:
        return 'skip'
    
    for i, (img_path, split) in enumerate(all_images, 1):
        try:
            current_decision = None
            
            img = mpimg.imread(img_path)
            fig, ax = plt.subplots(figsize=(10, 8))
            ax.imshow(img)
            ax.set_title(f"{species_name.upper()} ({split} {i}/{len(all_images)})\n\n"
                        f"C=Clean  |  X=Contaminated  |  U=Uncertain  |  N=Next  |  Q=Quit", 
                        fontsize=12)
            ax.axis('off')
            
            fig.canvas.mpl_connect('key_press_event', on_key)
            plt.tight_layout()
            plt.show(block=True)
            
            if current_decision == 'keep':
                return 'clean'
            elif current_decision == 'contaminated':
                return 'contaminated'
            elif current_decision == 'uncertain':
                return 'uncertain'
            elif current_decision == 'quit_all':
                return 'quit'
            # 'next' continues to next image
            
        except Exception as e:
            print(f"Error: {e}")
    
    # Saw all images without marking clean/contaminated
    return 'clean'  # Default to clean if user just pressed N through all

def main():
    print("="*70)
    print("QUICK CONTAMINATION SCANNER")
    print("="*70)
    print("Purpose: Identify which species need to be RE-SCRAPED")
    print("\nInstructions (press on image):")
    print("  C = Species is CLEAN (all images look correct)")
    print("  X = Species is CONTAMINATED (needs re-scrape)")
    print("  N = Next image (unsure, show me more)")
    print("  Q = Quit scanning")
    print("\n8 random images per species (6 train, 2 val)")
    print("="*70)
    
    train_dir = os.path.join(DATA_DIR, 'train')
    all_species = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\nFound {len(all_species)} species\n")
    
    input("Press Enter to start: ")
    
    results = {'clean': [], 'contaminated': [], 'uncertain': []}
    
    for idx, species in enumerate(all_species, 1):
        print(f"[{idx}/{len(all_species)}] {species}...", end=" ")
        
        status = scan_species(species)
        
        if status == 'quit':
            print("QUIT")
            break
        elif status == 'contaminated':
            results['contaminated'].append(species)
            print("❌ CONTAMINATED")
        elif status == 'uncertain':
            results['uncertain'].append(species)
            print("❓ UNCERTAIN")
        else:
            results['clean'].append(species)
            print("✓ clean")
        
        # Progress update every 10
        if idx % 10 == 0:
            print(f"\n--- Progress: {idx}/{len(all_species)} | Clean: {len(results['clean'])} | Contaminated: {len(results['contaminated'])} | Uncertain: {len(results['uncertain'])} ---\n")
    
    # Final report
    print("\n" + "="*70)
    print("SCAN COMPLETE")
    print("="*70)
    print(f"Clean: {len(results['clean'])}")
    print(f"Contaminated: {len(results['contaminated'])}")
    
    if results['contaminated']:
        print("\n🚨 SPECIES TO RE-SCRAPE:")
        for species in results['contaminated']:
            print(f"  - {species}")
    
    # Save report
    report = {
        'clean': results['clean'],
        'contaminated': results['contaminated'],
        'total_scanned': len(results['clean']) + len(results['contaminated'])
    }
    
    with open('contamination_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📁 Report saved to: contamination_report.json")

if __name__ == '__main__':
    main()
