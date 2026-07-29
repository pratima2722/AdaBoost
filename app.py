import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string, jsonify

app = Flask(__name__)

# --- MODEL LOADING ---
# Attempts to load your AdaBoost model file.
MODEL_PATH = "AdaBoost.pkl"
model = None

if os.path.exists(MODEL_PATH):
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"Warning: {MODEL_PATH} not found. Please place it in the same directory.")

# --- BEAUTIFUL LAYOUT & STYLE CODE ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Predictive Engine Portal</title>
    <style>
        :root {
            --bg-gradient: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            --card-bg: rgba(30, 41, 59, 0.7);
            --neon-cyan: #06b6d4;
            --neon-purple: #a855f7;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }

        body {
            background: var(--bg-gradient);
            color: var(--text-main);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 2rem 1rem;
            overflow-x: hidden;
        }

        /* Subtle Background Floating Animation */
        body::before {
            content: '';
            position: absolute;
            width: 300px;
            height: 300px;
            background: var(--neon-cyan);
            border-radius: 50%;
            filter: blur(120px);
            opacity: 0.15;
            top: 10%;
            left: 15%;
            animation: float 8s ease-in-out infinite alternate;
            z-index: -1;
        }

        @keyframes float {
            0% { transform: translateY(0px) scale(1); }
            100% { transform: translateY(40px) scale(1.1); }
        }

        .container {
            width: 100%;
            max-width: 850px;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 24px;
            padding: 3rem;
            box-shadow: 0 20px 40px rgba(0,0,0,0.4), 
                        0 0 50px rgba(6, 182, 212, 0.1);
            transform: translateY(20px);
            opacity: 0;
            animation: fadeInCard 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }

        @keyframes fadeInCard {
            to { transform: translateY(0); opacity: 1; }
        }

        header {
            text-align: center;
            margin-bottom: 2.5rem;
        }

        header h1 {
            font-size: 2.25rem;
            font-weight: 700;
            background: linear-gradient(to right, var(--neon-cyan), var(--neon-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }

        header p {
            color: var(--text-muted);
            font-size: 0.95rem;
        }

        /* Responsive Form Grid Matrix */
        .grid-form {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 1.5rem 2rem;
        }

        @media (max-width: 680px) {
            .grid-form { grid-template-columns: 1fr; }
            .container { padding: 1.5rem; }
        }

        .input-group {
            display: flex;
            flex-direction: column;
            position: relative;
        }

        .input-group label {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
            margin-bottom: 0.5rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            transition: color 0.3s ease;
        }

        .input-group input, .input-group select {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            color: var(--text-main);
            font-size: 1rem;
            outline: none;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .input-group input:focus, .input-group select:focus {
            border-color: var(--neon-cyan);
            box-shadow: 0 0 12px rgba(6, 182, 212, 0.3);
            background: rgba(15, 23, 42, 0.8);
        }

        .input-group input:focus + label {
            color: var(--neon-cyan);
        }

        .btn-container {
            grid-column: 1 / -1;
            margin-top: 1.5rem;
            display: flex;
            justify-content: center;
        }

        button {
            background: linear-gradient(90deg, #06b6d4, #3b82f6);
            color: white;
            border: none;
            border-radius: 12px;
            padding: 1rem 3rem;
            font-size: 1.1rem;
            font-weight: 600;
            cursor: pointer;
            box-shadow: 0 4px 14px rgba(6, 182, 212, 0.4);
            transition: all 0.3s ease;
            position: relative;
            overflow: hidden;
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(6, 182, 212, 0.6);
            background: linear-gradient(90deg, #22d3ee, #60a5fa);
        }

        button:active { transform: translateY(1px); }

        /* Elegant Dynamic Result Display */
        .result-display {
            margin-top: 2.5rem;
            padding: 1.5rem;
            border-radius: 16px;
            text-align: center;
            font-size: 1.25rem;
            font-weight: 600;
            display: none;
            border: 1px solid transparent;
            animation: popIn 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        }

        @keyframes popIn {
            0% { transform: scale(0.9); opacity: 0; }
            100% { transform: scale(1); opacity: 1; }
        }

        .result-success {
            background: rgba(16, 185, 129, 0.15);
            border-color: rgba(16, 185, 129, 0.3);
            color: #34d399;
        }

        .result-danger {
            background: rgba(239, 68, 68, 0.15);
            border-color: rgba(239, 68, 68, 0.3);
            color: #f87171;
        }
    </style>
</head>
<body>

<div class="container">
    <header>
        <h1>AdaBoost Analytics Portal</h1>
        <p>Enter individual telemetry metrics below to compute model diagnostics</p>
    </header>

    <form id="predictionForm" class="grid-form">
        <!-- Age -->
        <div class="input-group">
            <label for="age">Age</label>
            <input type="number" id="age" name="age" step="any" min="0" required placeholder="e.g. 34">
        </div>

        <!-- Gender -->
        <div class="input-group">
            <label for="gender">Gender</label>
            <select id="gender" name="gender" required>
                <option value="" disabled selected>Select Gender</option>
                <option value="1">Male</option>
                <option value="0">Female</option>
            </select>
        </div>

        <!-- Tenure -->
        <div class="input-group">
            <label for="tenure">Tenure (Months)</label>
            <input type="number" id="tenure" name="tenure" step="any" min="0" required placeholder="e.g. 12">
        </div>

        <!-- Usage Frequency -->
        <div class="input-group">
            <label for="usage_frequency">Usage Frequency</label>
            <input type="number" id="usage_frequency" name="usage_frequency" step="any" min="0" required placeholder="e.g. 20">
        </div>

        <!-- Support Calls -->
        <div class="input-group">
            <label for="support_calls">Support Calls</label>
            <input type="number" id="support_calls" name="support_calls" step="any" min="0" required placeholder="e.g. 2">
        </div>

        <!-- Payment Delay -->
        <div class="input-group">
            <label for="payment_delay">Payment Delay (Days)</label>
            <input type="number" id="payment_delay" name="payment_delay" step="any" min="0" required placeholder="e.g. 4">
        </div>

        <!-- Subscription Type -->
        <div class="input-group">
            <label for="subscription_type">Subscription Type</label>
            <select id="subscription_type" name="subscription_type" required>
                <option value="" disabled selected>Select Level</option>
                <option value="0">Basic</option>
                <option value="1">Standard</option>
                <option value="2">Premium</option>
            </select>
        </div>

        <!-- Contract Length -->
        <div class="input-group">
            <label for="contract_length">Contract Length</label>
            <select id="contract_length" name="contract_length" required>
                <option value="" disabled selected>Select Type</option>
                <option value="0">Month-to-Month</option>
                <option value="1">Annual</option>
                <option value="2">Two-Year</option>
            </select>
        </div>

        <!-- Total Spend -->
        <div class="input-group">
            <label for="total_spend">Total Spend</label>
            <input type="number" id="total_spend" name="total_spend" step="any" min="0" required placeholder="e.g. 450.50">
        </div>

        <!-- Last Interaction -->
        <div class="input-group">
            <label for="last_interaction">Last Interaction (Days ago)</label>
            <input type="number" id="last_interaction" name="last_interaction" step="any" min="0" required placeholder="e.g. 5">
        </div>

        <div class="btn-container">
            <button type="submit">Run Diagnostics</button>
        </div>
    </form>

    <div id="result" class="result-display"></div>
</div>

<script>
    document.getElementById('predictionForm').addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const resultDiv = document.getElementById('result');
        resultDiv.style.display = 'none';
        
        const formData = new FormData(this);
        const data = Object.fromEntries(formData.entries());
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data)
            });
            
            const resData = await response.json();
            
            if (response.ok) {
                resultDiv.innerText = `Prediction Verdict: Class ${resData.prediction}`;
                resultDiv.className = 'result-display ' + 
                    (resData.prediction === 1 ? 'result-success' : 'result-danger');
            } else {
                resultDiv.innerText = `Error: ${resData.error}`;
                resultDiv.className = 'result-display result-danger';
            }
            resultDiv.style.display = 'block';
        } catch (err) {
            resultDiv.innerText = 'Server down or network failure.';
            resultDiv.className = 'result-display result-danger';
            resultDiv.style.display = 'block';
        }
    });
</script>
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({'error': 'Machine learning model architecture file not found on server.'}), 500
        
    try:
        data = request.json
        
        # Mapping out all 10 verified keys in order
        feature_vector = [
            float(data['age']),
            float(data['gender']),
            float(data['tenure']),
            float(data['usage_frequency']),
            float(data['support_calls']),
            float(data['payment_delay']),
            float(data['subscription_type']),
            float(data['contract_length']),
            float(data['total_spend']),
            float(data['last_interaction'])
        ]
        
        # Reshaping vector to match scikit-learn expected format (1, 10)
        prediction = model.predict(np.array([feature_vector]))
        
        return jsonify({'prediction': int(prediction[0])})
        
    except KeyError as k_err:
        return jsonify({'error': f'Missing value entry field parameter: {str(k_err)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
