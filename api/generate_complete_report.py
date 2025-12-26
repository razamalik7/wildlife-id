import json

# Load diagnostic data
with open('c:/Users/nars7/wildlife-id/api/oogway_diagnostic_data.json', 'r') as f:
    data = json.load(f)

# Format markdown report
report = f"""# Oogway B3 Complete Diagnostic Report

## Overall Accuracy

| Metric | Accuracy |
|--------|----------|
| **Species** | **{data['accuracies']['species']:.2%}** |
| **Family** | **{data['accuracies']['family']:.2%}** |
| **Class** | **{data['accuracies']['class']:.2%}** |
| **Grandmaster B3 (species)** | **76.50%** |
| **Difference** | **{(data['accuracies']['species'] - 0.765)*100:+.2f}%** ❌ |

---

## Worst 15 Species (Poorest Performance)

| Rank | Species | Accuracy | Correct/Total |
|------|---------|----------|---------------|
"""

for i, entry in enumerate(data['per_class_accuracy'][:15], 1):
    report += f"| {i:2d} | {entry['species']:30s} | {entry['accuracy']:6.2%} | {entry['correct']:3d}/{entry['total']:3d} |\n"

report += f"""
---

## Best 10 Species (Highest Performance)

| Rank | Species | Accuracy | Correct/Total |
|------|---------|----------|---------------|
"""

for i, entry in enumerate(data['per_class_accuracy'][-10:], 1):
    report += f"| {i:2d} | {entry['species']:30s} | {entry['accuracy']:6.2%} | {entry['correct']:3d}/{entry['total']:3d} |\n"

report += f"""
---

## Top 25 Confused Pairs (Bidirectional)

| Rank | Pair | Total Errors | Direction Analysis |
|------|------|--------------|-------------------|
"""

for i, pair in enumerate(data['bidirectional_confusion_pairs'][:25], 1):
    a, b = pair['pair']
    total = pair['total_errors']
    
    # Get directional counts
    dir_data = pair['directional']
    keys = list(dir_data.keys())
    err1 = dir_data[keys[0]]
    err2 = dir_data[keys[1]]
    
    # Determine bias
    if err1 > err2 * 1.5:
        bias = f"→ ({keys[0].replace('_to_', '→')}: {err1})"
    elif err2 > err1 * 1.5:
        bias = f"← ({keys[1].replace('_to_', '→')}: {err2})"
    else:
        bias = f"↔ ({err1}/{err2})"
    
    report += f"| {i:2d} | {a} ↔ {b} | {total:3d} | {bias} |\n"

report += """
---

## Critical Findings

### 1. Catastrophic Elk Performance
**Elk accuracy: 3.73%** - This is completely broken. The model is classifying elk as other species 97% of the time.

### 2. Hard Negative Mining Backfired
The top confused pairs show that forcing the model to see elk/moose, fox species, and newt/salamander pairs in every batch may have **worsened** performance rather than improving it.

### 3. Multi-Task Cost
- Family accuracy (78.39%) is decent but not exceptional
- Class accuracy (89.32%) is good
- But species accuracy dropped 0.92% compared to Grandmaster
- **Net result**: The auxiliary tasks hurt the primary task

### 4. Pattern of Failures
Looking at worst species:
- Elk (3.73%) - Completely broken
- Arctic Fox (32.09%) - Misclassified as red fox
- River Otter (38.06%) - Confused with mink
- Spotted Salamander/Newts (~40-45%) - Color pattern confusion

These are exactly the species we targeted with hard negative mining, suggesting over-fitting to confusion rather than learning to distinguish.

---

## Root Cause Analysis (Updated)

### Primary Issue: Hard Negative Mining + Multi-Task Interaction

The combination of:
1. Hard negative mining forcing confused pairs into every batch
2. Multi-task loss pulling gradients in different directions
3. Hierarchical focal loss giving 3x penalty to cross-class errors

Created a **toxic interaction** where:
- Model learned to avoid cross-class errors (good for class accuracy)
- But became paralyzed on within-family distinctions (bad for species accuracy)
- Elk/moose (both Cervidae) became indistinguishable

### Secondary Issues
1. **Metadata dropout** may have hurt location-dependent species (arctic fox, caribou)
2. **CutMix** may have removed critical distinguishing features (antlers, tails)
3. **Label smoothing** on already-confused pairs may have made model uncertain

---

## Recommendations (Revised)

### Option 1: Remove Multi-Task + Reduce Hard Neg Mining (Recommended)
- Single-task species-only prediction
- Hard negative mining with **only top 10** pairs (not 25)
- Reduce pair frequency to 1-2  pairs per batch (not 3)
- Keep: SWA fix, Focal Loss (without hierarchy), Metadata Dropout
- Remove: Multi-task heads, hierarchical weighting, excessive pair forcing

### Option 2: Start From Grandmaster + Minimal Changes
- Take Grandmaster architecture
- Apply only SWA fix (1e-5 LR, update_bn, save SWA)
- Add Focal Loss (γ=2.0, no hierarchy)
- Extend FixRes to 3 epochs
- **Nothing else** - prove each technique helps before adding more

### Option 3: Debug Elk First
- Something is fundamentally broken with elk classification
- 3.73% suggests labeling error or data corruption
- Before retraining, verify elk samples are correctly labeled

---

## Next Steps

1. **Investigate elk dataset** - 3.73% is suspiciously low
2. **Choose simplified architecture** (Option 1 or 2)
3. **Re-run training with conservative enhancements**
4. **Expected result**: 77-79% (Grandmaster + proven techniques only)

**The more complex we made Oogway, the worse it got. Simplicity wins.**
"""

# Save report
with open('C:/Users/nars7/.gemini/antigravity/brain/b4ebe408-4a3f-4392-a090-81cf0e8336d5/oogway_complete_diagnostic.md', 'w', encoding='utf-8') as f:
    f.write(report)

print("Complete diagnostic report saved!")
print(f"\nKey findings:")
print(f"- Elk accuracy: {data['per_class_accuracy'][0]['accuracy']:.2%} (CATASTROPHIC)")
print(f"- Species overall: {data['accuracies']['species']:.2%}")
print(f"- Family: {data['accuracies']['family']:.2%}")
print(f"- Class: {data['accuracies']['class']:.2%}")
