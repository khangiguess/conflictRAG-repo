import json

def count_jsonl_samples(file_path):
    sample_count = 0
    invalid_count = 0
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line_number, line in enumerate(file, start=1):
            # Skip perfectly empty lines
            if not line.strip():
                continue
                
            try:
                # Parse the JSON line
                data = json.loads(line)
                
                # Verify the expected keys are present
                if "input" in data and "target" in data:
                    sample_count += 1
                else:
                    print(f"Warning: Line {line_number} is missing 'input' or 'target' keys.")
                    invalid_count += 1
                    
            except json.JSONDecodeError:
                print(f"Error: Invalid JSON format on line {line_number}")
                invalid_count += 1
                
    print(f"Successfully counted {sample_count} valid samples.")
    if invalid_count > 0:
        print(f"Found {invalid_count} invalid lines.")
        
    return sample_count

# --- Example Usage ---
# Replace 'dataset.jsonl' with the actual path to your file
if __name__ == "__main__":
    file_path = 't5_training_data/test.jsonl' 
    total_samples = count_jsonl_samples(file_path)