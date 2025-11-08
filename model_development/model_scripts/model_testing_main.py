import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import numpy as np
import random
import os

# =============================================================================
# SECTION 1: DEFINE YOUR *EXACT* MODEL ARCHITECTURE
# This MUST match the class from your training script.
# =============================================================================

class AccidentDetectionModel(nn.Module):
    """CNN-LSTM model for accident detection"""

    def __init__(self, input_features=10, hidden_dim=128, num_layers=2, dropout=0.3, use_attention=True):
        super(AccidentDetectionModel, self).__init__()

        self.use_attention = use_attention

        # CNN layers
        self.conv1 = nn.Conv1d(input_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(2)

        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)

        # LSTM
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim * 2, 128),
                nn.Tanh(),
                nn.Linear(128, 1)
            )

        # Classifier
        lstm_output_dim = hidden_dim * 2
        self.fc1 = nn.Linear(lstm_output_dim, 256)
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(dropout)
        self.fc3 = nn.Linear(128, 2) # 2 classes: Non-Accident, Accident

    def forward(self, x):
        # x: (batch, timesteps, features) -> [1, 150, 10]
        x = x.transpose(1, 2)  # (batch, features, timesteps) -> [1, 10, 150]

        # CNN
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x))) # Shape: [1, 256, 37]

        x = x.transpose(1, 2)  # (batch, timesteps, features) -> [1, 37, 256]

        # LSTM
        lstm_out, _ = self.lstm(x) # Shape: [1, 37, 256]

        # Attention
        if self.use_attention:
            attention_weights = self.attention(lstm_out)
            attention_weights = F.softmax(attention_weights, dim=1)
            context = torch.sum(attention_weights * lstm_out, dim=1) # Shape: [1, 256]
        else:
            context = lstm_out[:, -1, :] # Shape: [1, 256]

        # Classifier
        x = F.relu(self.fc1(context))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x) # Shape: [1, 2] (Logits)

        return x

print("✅ Model architecture defined.")


# =============================================================================
# SECTION 2: TEST DATA GENERATION & NORMALIZATION
# =============================================================================

# --- Configuration (Must match training) ---
WINDOW_SIZE = 150  # 3 seconds * 50Hz
NUM_FEATURES = 10  # ax, ay, az, gx, gy, gz, mx, my, mz, spd
FEATURES_ORDER = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz', 'spd']
SAMPLE_RATE = 50 # from your training script

def generate_raw_accident_sample():
    """
    Generates a *raw*, unnormalized sample in the same format
    as your training script (a list of dicts).
    This simulates a high-G-force event.
    
    --- THIS FUNCTION IS NOW A COPY OF YOUR 'generate_accident' TRAINING SCRIPT ---
    """
    print("Generating a raw (unnormalized) 'Accident' sample...")
    
    data_batch = []
    initial_speed = random.uniform(50, 120)
    crash_point = int(WINDOW_SIZE * 0.6) # 90th timestep
    base_mx, base_my, base_mz = random.uniform(40, 50), random.uniform(10, 20), random.uniform(-5, 5)
    accident_type='frontal' # Hardcode a frontal impact for the test
    
    print(f"Injecting high-G 'frontal' spike at timestep {crash_point}...")

    for i in range(WINDOW_SIZE):
        if i < crash_point:
            ax, ay, az = np.random.normal(0, 0.2), 9.8 + np.random.normal(0, 0.2), np.random.normal(0, 0.2)
            gx, gy, gz = np.random.normal(0, 0.05), np.random.normal(0, 0.05), np.random.normal(0, 0.02)
            spd = initial_speed + np.random.normal(0, 2)
            mx, my, mz = base_mx + np.random.normal(0, 0.5), base_my + np.random.normal(0, 0.5), base_mz + np.random.normal(0, 0.3)
        elif i < crash_point + 25:
            progress = (i - crash_point)
            intensity = np.exp(-progress / 8)

            if accident_type == 'frontal':
                ax = -random.uniform(6, 12) * intensity + np.random.normal(0, 2)
                ay = 9.8 + np.random.uniform(-3, 3) * intensity
                gy = random.uniform(1, 3) * intensity
            elif accident_type == 'side':
                ax = np.random.uniform(-3, 3) * intensity
                ay = 9.8 + random.uniform(-8, 8) * intensity
                gy = np.random.uniform(-1, 1) * intensity
            else:  # rear
                ax = random.uniform(4, 8) * intensity + np.random.normal(0, 2)
                ay = 9.8 + np.random.uniform(-2, 2) * intensity
                gy = random.uniform(-3, -1) * intensity

            az = np.random.uniform(-3, 3) * intensity
            gx = random.uniform(-2, 2) * intensity
            gz = np.random.uniform(-1, 1) * intensity
            
            # --- THIS IS THE CORRECTED SPEED LOGIC ---
            spd = initial_speed * (1 - progress / 25) * (0.3 + 0.7 * intensity) 
            
            mx, my, mz = base_mx + np.random.uniform(-5, 5) * intensity, base_my + np.random.uniform(-5, 5) * intensity, base_mz + np.random.uniform(-3, 3) * intensity
        else:
            ax, ay, az = np.random.normal(0, 0.3), 9.8 + np.random.normal(0, 0.4), np.random.normal(0, 0.3)
            gx, gy, gz = np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(0, 0.05)
            spd = max(0, np.random.uniform(0, 5))
            mx, my, mz = base_mx + np.random.normal(0, 1), base_my + np.random.normal(0, 1), base_mz + np.random.normal(0, 0.5)

        data_batch.append({
            "ax": round(ax, 4), "ay": round(ay, 4), "az": round(az, 4),
            "gx": round(gx, 4), "gy": round(gy, 4), "gz": round(gz, 4),
            "mx": round(mx, 4), "my": round(my, 4), "mz": round(mz, 4),
            "spd": round(max(0, spd), 2)
        })
    return data_batch


def normalize_sample(raw_sample_list, stats):
    """
    Normalizes the raw sample using the loaded stats.
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
    Runs the normalized data through the loaded original model.
    """
    print("Converting to tensor and running inference...")
    
    # 1. Convert numpy array to torch tensor
    # 2. Add a batch dimension (unsqueeze(0)) to match model input [1, 150, 10]
    input_tensor = torch.tensor(normalized_data).unsqueeze(0) 
    
    # 3. Run inference inside a no_grad() block
    with torch.no_grad():
        logits = model(input_tensor) 
    
    # 4. Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    return logits, probabilities

# =============================================================================
# SECTION 3: MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    CHECKPOINT_PATH = 'best_model.pth'
    STATS_PATH = 'normalization_stats.json'

    # Check if files exist
    if not (os.path.exists(CHECKPOINT_PATH) and os.path.exists(STATS_PATH)):
        print(f"Error: Missing files. Make sure '{CHECKPOINT_PATH}' and '{STATS_PATH}' are in this directory.")
    else:
        print("="*80)
        print("Starting Test for ORIGINAL .pth Model")
        print("="*80)
        
        # 1. Load Stats
        print(f"Loading stats from {STATS_PATH}...")
        with open(STATS_PATH, 'r') as f:
            stats = json.load(f)
        print(f"Loaded stats for {len(stats['features'])} features.")
        
        # 2. Load Model (The key difference is here)
        print(f"Loading model weights from {CHECKPOINT_PATH}...")
        
        # Instantiate the model class
        model = AccidentDetectionModel(
            input_features=NUM_FEATURES,
            hidden_dim=128,
            num_layers=2,
            dropout=0.3,
            use_attention=True
        )
        
        # Load the checkpoint
        # map_location='cpu' ensures it runs even without a GPU
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device('cpu'))
        
        # Load the weights into the model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set to evaluation mode (very important!)
        model.eval() 
        print("✅ Original PyTorch .pth model loaded successfully.")
        
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