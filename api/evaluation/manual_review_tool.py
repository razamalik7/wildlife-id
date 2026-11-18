"""
Comprehensive manual review tool for ALL species
Uses matplotlib keyboard shortcuts - press k/d directly on image window
"""
import os
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

DATA_DIR = 'training_data_cropped'
QUARANTINE_DIR = 'quarantine_manual'

# Global for capturing keyboard events
current_decision = None

def on_key(event):
    """Handle keyboard events on the matplotlib window"""
    global current_decision
    if event.key in ['k', 'enter']:
        current_decision = 'keep'
    elif event.key == 'd':
        current_decision = 'delete'
    elif event.key == 'q':
        current_decision = 'quit'
    plt.close()

def review_species(species_name):
    """Show sampled images for a species"""
    global current_decision
    
    print(f"\n{'='*70}")
    print(f"{species_name.upper()}")
    print(f"{'='*70}")
    
    to_delete = []
    
    # Sample 6 from train, 2 from val
    samples_per_split = {'train': 6, 'val': 2}
    
    for split, num_samples in samples_per_split.items():
        species_dir = os.path.join(DATA_DIR, split, species_name)
        
        if not os.path.exists(species_dir):
            continue
            
        all_images = sorted([f for f in os.listdir(species_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        images = random.sample(all_images, min(num_samples, len(all_images)))
        
        print(f"{split}: {len(images)} images", end=" ")
        
        for i, img_name in enumerate(images, 1):
            img_path = os.path.join(species_dir, img_name)
            
            try:
                current_decision = None
                
                img = mpimg.imread(img_path)
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.imshow(img)
                ax.set_title(f"{species_name} - {split} {i}/{len(images)}\nPress: K=keep, D=delete, Q=quit", 
                           fontsize=14, pad=20)
                ax.axis('off')
                
                # Connect keyboard event
                fig.canvas.mpl_connect('key_press_event', on_key)
                
                plt.tight_layout()
                plt.show(block=True)  # Block until user presses a key
                
                if current_decision == 'keep':
                    print("✓", end=" ")
                elif current_decision == 'delete':
                    to_delete.append(img_path)
                    print("✗", end=" ")
                elif current_decision == 'quit':
                    print("\nQuitting species")
                    return to_delete
                    
            except Exception as e:
                print(f"Error: {e}")
        
        print()  # New line after split
    
    return to_delete

def main():
    print("="*70)
    print("COMPREHENSIVE DATA REVIEW - ALL 100 SPECIES")
    print("="*70)
    print("6 train + 2 val = 8 images per species (~800 total)")
    print("\nInstructions:")
    print("  Press K on image = Keep")
    print("  Press D on image = Delete")
    print("  Press Q on image = Quit species")
    print("  (No need to click terminal!)")
    print("="*70)
    
    train_dir = os.path.join(DATA_DIR, 'train')
    all_species = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    
    print(f"\nFound {len(all_species)} species\n")
    
    input("Press Enter to start: ")
    
    all_marked = []
    
    for idx, species in enumerate(all_species, 1):
        print(f"\n[{idx}/{len(all_species)}] ", end="")
        marked = review_species(species)
        all_marked.extend(marked)
        
        if idx % 10 == 0:
            print(f"\n--- {idx}/{len(all_species)} done, {len(all_marked)} marked ---")
    
    # Summary
    print("\n" + "="*70)
    print(f"COMPLETE: {len(all_marked)} images marked")
    print("="*70)
    
    if all_marked:
        confirm = input(f"\nDELETE {len(all_marked)} images? (yes/no): ").strip().lower()
        
        if confirm == 'yes':
            os.makedirs(QUARANTINE_DIR, exist_ok=True)
            import shutil
            
            for img_path in all_marked:
                species_name = img_path.split(os.sep)[-3]
                split = img_path.split(os.sep)[-2]
                
                dest = os.path.join(QUARANTINE_DIR, species_name, split)
                os.makedirs(dest, exist_ok=True)
                
                shutil.move(img_path, os.path.join(dest, os.path.basename(img_path)))
            
            print(f"\n✓ Quarantined {len(all_marked)} images")
            
            # Show contamination summary
            from collections import defaultdict
            species_counts = defaultdict(int)
            for path in all_marked:
                species = path.split(os.sep)[-3]
                species_counts[species] += 1
            
            if species_counts:
                print("\nMost contaminated:")
                for species, count in sorted(species_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
                    print(f"  {species}: {count}")
        else:
            print("Cancelled")
    else:
        print("\nNo contamination found!")

if __name__ == '__main__':
    main()
