# Attention is Really All You Need: EEG-Enhanced Language Models

## 📑 Overview

This project implements an innovative brain-computer interface that enhances large language models by adapting their responses based on real-time EEG attention signals. The system uses a Mixture-of-Experts (MoE) approach to dynamically adjust the language model's generation process according to the user's cognitive state.

![System Architecture](https://i.imgur.com/QHCY1sf.png)

## 🧠 Key Features

- **Real-time EEG signal processing** - Processes brain signals to extract attention levels
- **Adaptive Language Model Generation** - Dynamically adjusts text generation based on attention levels
- **Mixture-of-Experts Controller** - Routes generation through different "expert" configurations based on cognitive state
- **Interactive Visualization** - Real-time visualization of attention levels and expert routing
- **Flexible Architecture** - Works with both simulated and real EEG data from Muse headsets

## 🔬 How It Works

The system consists of three main components:

1. **EEG Processor**: Analyzes EEG signals to extract attention metrics, either from a real Muse headset or through a realistic simulation.

2. **MoE Controller**: Implements a Mixture-of-Experts approach with three distinct experts:
   - **Simplicity Expert**: Optimized for low attention states - uses simpler language and clearer explanations
   - **Balanced Expert**: Optimized for medium attention - balances complexity and clarity
   - **Complexity Expert**: Optimized for high attention - provides more detailed technical information

3. **EEG-Enhanced LLM**: Integrates the DeepSeek-R1 language model with the EEG processor and MoE controller, enabling token-by-token generation influenced by attention signals.

During text generation, the system continuously monitors attention levels and adjusts generation parameters, including:
- Temperature (higher for low attention, lower for high attention)
- Top-k sampling parameters
- Repetition penalties
- Token biasing toward common or rare vocabulary
- Introduction of style tokens that match the cognitive state

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.18+
- NumPy
- Matplotlib
- (Optional) muselsl and pylsl for real EEG data

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/eeg-enhanced-llm.git
cd eeg-enhanced-llm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) For real EEG data with Muse headset:
```bash
pip install muselsl pylsl
```

## 🎮 Usage

### Running the Demo

The simplest way to experience the system is to run the interactive demo:

```bash
python demo.py
```

This will start an interactive session with simulated EEG data and visualization of attention levels and expert weights.

### Command Line Options

```
usage: demo.py [-h] [--model MODEL] [--real-eeg] [--no-visualization]

optional arguments:
  -h, --help       show this help message and exit
  --model MODEL    Path to the language model
  --real-eeg       Use real EEG data from Muse headset instead of simulation
  --no-visualization  Disable visualization
```

### Using Real EEG Data

To use real EEG data from a Muse headset:

1. Ensure your Muse headset is connected via Bluetooth
2. Run the demo with the `--real-eeg` flag:

```bash
python demo.py --real-eeg
```

## 📊 Technical Details

### EEG Signal Processing

- The system extracts attention metrics by analyzing the balance between alpha (8-13 Hz) and beta (13-30 Hz) waves
- Higher beta/alpha ratio generally indicates higher cognitive engagement
- A sigmoid normalization function maps raw values to a 0-1 attention scale
- Exponential moving average smoothing reduces noise and prevents rapid fluctuations

### MoE Controller Architecture

The MoE controller implements three expert configurations:

1. **Simplicity Expert (Low Attention)**
   - Higher temperature (1.1) for more varied, simpler outputs
   - Lower top-k (20) to focus on common tokens
   - Boosts frequency of common words and simplifying phrases

2. **Balanced Expert (Medium Attention)**
   - Moderate temperature (0.9) 
   - Moderate top-k (40)
   - Balanced token distribution without strong biasing

3. **Complexity Expert (High Attention)**
   - Lower temperature (0.7) for more deterministic, focused outputs
   - Higher top-k (60) allowing for more specialized tokens
   - Boosts frequency of rare/technical tokens and complex phrases

Expert weights are calculated using a softmax-like approach based on the current attention level, with smooth transitions between states.

### Language Model Integration

The system implements token-by-token generation with EEG-based steering:

1. For each generation step, the current attention level is sampled
2. MoE controller weights are updated based on attention
3. Logits from the language model are modified according to expert weights
4. Temperature and sampling parameters are dynamically adjusted
5. Generation statistics are tracked for analysis

## 📝 Project Structure

```
├── main_eeg.py            # Main EEG-enhanced LLM implementation
├── eeg_processor.py       # EEG signal processing module
├── moe_control.py         # Mixture-of-Experts controller
├── demo.py                # Interactive demonstration script
└── requirements.txt       # Project dependencies
```

## 🔗 Future Work

- Integration with more advanced EEG devices and signal processing techniques
- Implementation of more sophisticated MoE architectures with specialized experts
- Fine-tuning of language models on EEG-correlated text data
- Development of personalized attention-response mappings
- Exploration of other cognitive signals beyond attention

## 📄 License

MIT License

## 🙏 Acknowledgements

- DeepSeek AI for the DeepSeek-R1 model
- Interaxon for the Muse EEG headset technology
- The open-source NLP and BCI communities