import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.quantization
from torch.utils.mobile_optimizer import optimize_for_mobile

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
# SECTION 2: LOAD, QUANTIZE, AND CONVERT
# =============================================================================

# --- 2a. Set Parameters (Must match training) ---
INPUT_FEATURES = 10  # ax, ay, az, gx, gy, gz, mx, my, mz, spd
NUM_CLASSES = 2    # Non-Accident, Accident
WINDOW_SIZE = 150  # 3 seconds * 50Hz

# --- 2b. Load Your Trained Weights ---
# Initialize the model structure
model = AccidentDetectionModel(
    input_features=INPUT_FEATURES,
    hidden_dim=128,
    num_layers=2,
    dropout=0.3,
    use_attention=True
)

# Load the saved checkpoint from 'best_model.pth'
# Your script saved a checkpoint dictionary, so we load 'model_state_dict'
try:
    checkpoint = torch.load('best_model.pth', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode (disables dropout, etc.)
    print("✅ Model weights loaded successfully from best_model.pth")
except Exception as e:
    print(f"❌ ERROR: Could not load weights. Make sure 'best_model.pth' is in the same directory.")
    print(f"Details: {e}")
    # Stop execution if loading fails
    raise

# --- 2c. Apply Dynamic Quantization ---
print("Applying Dynamic Quantization...")
# This quantizes the weights of the Linear and LSTM layers (the heaviest parts)
# to int8, which significantly reduces model size and speeds up CPU inference.
quantized_model = torch.quantization.quantize_dynamic(
    model, 
    {nn.LSTM, nn.Linear}, # Specify layers to quantize
    dtype=torch.qint8
)
print("✅ Dynamic Quantization applied.")

# --- 2d. Trace and Convert the Quantized Model ---
print("Tracing model for mobile...")
# Create a dummy input tensor with the correct shape:
# [batch_size, sequence_length, num_features]
dummy_input = torch.rand(1, WINDOW_SIZE, INPUT_FEATURES)

# Trace the *quantized* model
traced_model = torch.jit.trace(quantized_model, dummy_input)

# Optimize the traced model for mobile
optimized_model = optimize_for_mobile(traced_model)

# --- 2e. Save the Final Mobile Model ---
# This is the *only* file you need to deploy to the React Native app.
output_path = "best_model.ptl"
optimized_model._save_for_lite_interpreter(output_path)

print("\n" + "="*80)
print(f"✅ Model successfully quantized, converted, and saved to: {output_path}")
print("You can now copy 'best_model.ptl' and 'normalization_stats.json' to your React Native project assets.")
print("="*80)