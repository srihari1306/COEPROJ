import torch
import json
import numpy as np
import random
import os

# --- 1. Configuration (MUST MATCH YOUR FILES) ---
MODEL_PATH = 'best_model.ptl'
STATS_PATH = 'normalization_stats.json'
WINDOW_SIZE = 150  # 3 seconds * 50Hz
NUM_FEATURES = 10  # ax, ay, az, gx, gy, gz, mx, my, mz, spd
FEATURES_ORDER = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'spd']

def generate_raw_accident_sample():
    """
    Generates a *raw*, unnormalized sample in the same format
    as your training script (a list of dicts).
    This simulates a high-G-force event.
    """
    print("Generating a raw (unnormalized) 'Accident' sample...")
    sample = []
    
    # Generate 150 timesteps (3 seconds)
    for i in range(WINDOW_SIZE):
        # Start with a normal driving baseline
        step = {
            'ax': round(np.random.normal(0, 0.15), 4),
            'ay': round(9.8 + np.random.normal(0, 0.2), 4),
            'az': round(np.random.normal(0, 0.15), 4),
            'gx': round(np.random.normal(0, 0.05), 4),
            'gy': round(np.random.normal(0, 0.05), 4),
            'gz': round(np.random.normal(0, 0.02), 4),
            'mx': round(45.0 + np.random.normal(0, 0.5), 4),
            'my': round(15.0 + np.random.normal(0, 0.5), 4),
            'mz': round(-2.0 + np.random.normal(0, 0.3), 4),
            'spd': round(max(0, 80.0 + np.random.normal(0, 2)), 2)
        }
        sample.append(step)

    # Inject the accident spike (matching your 'generate_accident' logic)
    # This spike is RAW and has NOT been normalized.
    crash_point = int(WINDOW_SIZE * 0.6) # 90th timestep
    print(f"Injecting high-G spike at timestep {crash_point}...")
    
    initial_speed = 80.0 # Match the baseline
    for i in range(crash_point, crash_point + 25): # Match training script duration
        if i < WINDOW_SIZE:
            progress = (i - crash_point)
            intensity = np.exp(-progress / 8)
            
            # Simulate a strong frontal impact
            sample[i]['ax'] = round(-random.uniform(6, 12) * intensity + np.random.normal(0, 2), 4)
            sample[i]['ay'] = round(9.8 + np.random.uniform(-3, 3) * intensity, 4)
            sample[i]['gy'] = round(random.uniform(1, 3) * intensity, 4)
            
            # Use the EXACT speed logic from training
            spd = initial_speed * (1 - progress / 25) * (0.3 + 0.7 * intensity)
            sample[i]['spd'] = round(max(0, spd), 2)
            
            # Add other noise as per training
            sample[i]['az'] = round(np.random.uniform(-3, 3) * intensity, 4)
            sample[i]['gx'] = round(random.uniform(-2, 2) * intensity, 4)
            sample[i]['gz'] = round(np.random.uniform(-1, 1) * intensity, 4)
            sample[i]['mx'] = round(45.0 + np.random.uniform(-5, 5) * intensity, 4)
            sample[i]['my'] = round(15.0 + np.random.uniform(-5, 5) * intensity, 4)
            sample[i]['mz'] = round(-2.0 + np.random.uniform(-3, 3) * intensity, 4)

    return sample

def normalize_sample(raw_sample_list, stats):
    """
    Normalizes the raw sample using the loaded stats.
    This is the *exact* logic your React Native app must implement.
    """
    print("Normalizing raw data using normalization_stats.json...")
    normalized_array = np.zeros((WINDOW_SIZE, NUM_FEATURES), dtype=np.float32)
    
    for t in range(WINDOW_SIZE):
        for i, key in enumerate(FEATURES_ORDER):
            raw_val = raw_sample_list[t][key]
            mean = stats['mean'][key]
            std = stats['std'][key]
            normalized_array[t, i] = (raw_val - mean) / std
            
    print(f"Sample shape after normalization: {normalized_array.shape}")
    return normalized_array

def run_inference(model, normalized_data):
    """
    Runs the normalized data through the loaded mobile model.
    """
    print("Converting to tensor and running inference...")
    
    # 1. Convert numpy array to torch tensor
    # 2. Add a batch dimension (unsqueeze(0)) to match model input [1, 150, 10]
    input_tensor = torch.tensor(normalized_data).unsqueeze(0) 
    
    # 3. Run inference inside a no_grad() block
    with torch.no_grad():
        # model.forward(input_tensor)
        logits = model(input_tensor) 
    
    # 4. Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    return logits, probabilities

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    
    # Check if files exist
    if not (os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)):
        print(f"Error: Missing files. Make sure '{MODEL_PATH}' and '{STATS_PATH}' are in this directory.")
    else:
        print("="*80)
        print("Starting Test for Deployed Mobile Model")
        print("="*80)
        
        # 1. Load Stats
        print(f"Loading stats from {STATS_PATH}...")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats for {len(stats['features'])} features.")
        
        # 2. Load Model
        print(f"Loading mobile model from {MODEL_PATH}...")
        # Use torch.jit.load() for .ptl or .pt files
        model = torch.jit.load(MODEL_PATH)
        model.eval() # Set to evaluation mode
        print("✅ Model loaded successfully.")
        
        # 3. Get Raw Data
        raw_accident_sample = generate_raw_accident_sample()
        
        # 4. Normalize Data
        normalized_accident_sample = normalize_sample(raw_accident_sample, stats)
        
        # 5. Run Inference
        logits, probabilities = run_inference(model, normalized_accident_sample)
        
        # 6. Show Results
        probs_list = probabilities.numpy()[0]
        prediction = np.argmax(probs_list)
        
        print("\n" + "="*80)
        print("INFERENCE RESULTS")
        print("="*80)
        print(f"Raw Logits: {logits.numpy()[0]}")
        print(f"  P(Non-Accident) [0]: {probs_list[0] * 100:.2f}%")
        print(f"  P(Accident)     [1]: {probs_list[1] * 100:.2f}%")
        
        print("\n--- FINAL PREDICTION ---")
        if prediction == 1 and probs_list[1] > 0.8:
            print(f"✅ SUCCESS: Model correctly predicted 'Accident' with high confidence ({probs_list[1]*100:.2f}%)")
        else:
            print(f"⚠️ FAILED: Model did not confidently predict 'Accident'.")
            
        print("="*80)