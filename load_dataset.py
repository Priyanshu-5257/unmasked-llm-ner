import os
import glob
import pandas as pd

def parse_brat_annotations(data_dir):
    dataset = []
    # Find all text files in the target directory
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    for txt_path in txt_files:
        base_path = os.path.splitext(txt_path)[0]
        a1_path = base_path + ".a1"  # Contains the core NER entities
        
        # Read raw text sequence
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        entities = []
        if os.path.exists(a1_path):
            with open(a1_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or not line.startswith('T'):
                        continue  # Skip event relations, target text-bound entities only
                    
                    try:
                        # BRAT format: ID \t Label Start End \t Text
                        parts = line.split('\t')
                        entity_id = parts[0]
                        meta = parts[1].split(' ')
                        
                        entities.append({
                            "entity_id": entity_id,
                            "label": meta[0],
                            "start": int(meta[1]),
                            "end": int(meta[2]),
                            "text": parts[2]
                        })
                    except (IndexError, ValueError):
                        continue
                        
        dataset.append({
            "file_id": os.path.basename(base_path),
            "text": text,
            "entities": entities
        })
        
    return dataset

# Example execution pointing to your extracted archive path
train_set = parse_brat_annotations("./BioNLP-ST_2013_CG_train_data")
print(f"Successfully processed {len(train_set)} documents locally.")
