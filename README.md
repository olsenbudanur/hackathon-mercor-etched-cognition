# EEG-Enhanced Language Model with Mixture-of-Experts

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This project implements a brain-computer interface that enhances language model generation based on real-time EEG signals. The system uses a Mixture-of-Experts (MoE) approach to dynamically adapt the model's output based on detected attention levels.

## 📑 Overview

This is an innovative brain-computer interface that enhances large language models by adapting their responses based on real-time EEG attention signals. The system uses a Mixture-of-Experts (MoE) approach to dynamically adjust the language model's generation process according to the user's cognitive state.

## 🧠 Key Features

- **Real-time EEG signal processing** - Processes brain signals to extract attention levels
- **Adaptive Language Model Generation** - Dynamically adjusts text generation based on attention levels
- **Mixture-of-Experts Controller** - Routes generation through different "expert" configurations based on cognitive state
- **Interactive Visualization** - Real-time visualization of attention levels and expert routing
- **Flexible Architecture** - Works with both simulated and real EEG data from Muse headsets

## 🔬 How It Works

The system consists of four main components:

1. **Backend (FastAPI)**: Handles token streaming and expert routing
2. **Frontend (React)**: Visualizes the streamed tokens with different colors based on the expert
3. **EEG Processing**: Processes EEG signals to extract attention metrics
4. **Kevin-MOE Demo**: Implements the EEG-Enhanced Language Model with Mixture-of-Experts

During text generation, the system continuously monitors attention levels and adjusts generation parameters, including:
- Temperature (higher for low attention, lower for high attention)
- Top-k sampling parameters
- Repetition penalties
- Token biasing toward common or rare vocabulary
- Introduction of style tokens that match the cognitive state

## 📂 Repository Structure

```
hackathon-mercor-etched-cognition/
├── backend/                # FastAPI backend for token streaming
│   └── main.py             # Main FastAPI application
├── frontend/               # React frontend for visualization
│   ├── public/             # Public assets
│   ├── src/                # React source code
│   └── package.json        # Frontend dependencies
├── eeg/                    # EEG processing scripts
│   ├── eeg.ipynb           # Jupyter notebook for EEG processing
│   └── eeg.py              # Python script for EEG processing
├── kevin-moe-demo/         # Kevin-MOE demo implementation
│   ├── main_eeg.py         # Core EEG-enhanced LLM implementation
│   ├── eeg_processor.py    # EEG signal processing module
│   ├── moe_control.py      # Mixture-of-Experts controller
│   └── demo.py             # Interactive demonstration script
├── tests/                  # Test suite
│   ├── backend/            # Backend tests
│   ├── eeg/                # EEG processing tests
│   ├── frontend/           # Frontend tests
│   └── integration/        # Integration tests
├── docs/                   # Documentation
│   ├── installation.md     # Installation instructions
│   ├── usage.md            # Usage instructions
│   ├── api.md              # API documentation
│   ├── contributing.md     # Contributing guidelines
│   └── technical_approach.md # Technical approach documentation
├── LICENSE                 # MIT License
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── setup.py                # Package installation
├── Makefile                # Common tasks
├── Dockerfile              # Docker configuration
└── docker-compose.yml      # Docker Compose configuration
```

## 🛠️ Installation

See [Installation Guide](docs/installation.md) for detailed instructions.

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/yourusername/hackathon-mercor-etched-cognition.git
cd hackathon-mercor-etched-cognition
```

2. Install backend dependencies:
```bash
pip3 install fastapi uvicorn
pip3 install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd frontend
npm install
cd ..
```

4. Start the backend:
```bash
cd backend
uvicorn main:app --reload
```

5. Start the frontend (in a new terminal):
```bash
cd frontend
npm start
```

## 🎮 Usage

See [Usage Guide](docs/usage.md) for detailed instructions.

### Basic Demo

```bash
cd kevin-moe-demo
python3 demo.py
```

### Using Real EEG Data

```bash
cd kevin-moe-demo
python3 demo.py --real-eeg
```

## 📊 Technical Approaches

See [Technical Approach](docs/technical_approach.md) for detailed information.

### EEG Signal Processing

The system extracts attention metrics by analyzing the balance between alpha (8-13 Hz) and beta (13-30 Hz) waves. Higher beta/alpha ratio generally indicates higher cognitive engagement.

### Mixture-of-Experts Controller

The MoE controller implements three expert configurations:

1. **Simplicity Expert (Low Attention)**
   - Higher temperature for more varied, simpler outputs
   - Lower top-k to focus on common tokens
   - Boosts frequency of common words and simplifying phrases

2. **Balanced Expert (Medium Attention)**
   - Moderate temperature
   - Moderate top-k
   - Balanced token distribution without strong biasing

3. **Complexity Expert (High Attention)**
   - Lower temperature for more deterministic, focused outputs
   - Higher top-k allowing for more specialized tokens
   - Boosts frequency of rare/technical tokens and complex phrases

### Token Streaming

The system implements token-by-token generation with EEG-based steering:

1. For each generation step, the current attention level is sampled
2. MoE controller weights are updated based on attention
3. Logits from the language model are modified according to expert weights
4. Temperature and sampling parameters are dynamically adjusted
5. Generation statistics are tracked for analysis

## 🧪 Testing

```bash
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](docs/contributing.md) for more information.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgements

- The open-source NLP and BCI communities
- Contributors to the FastAPI, React, and PyTorch ecosystems
