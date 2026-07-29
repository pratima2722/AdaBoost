import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# Load the AdaBoost model safely
MODEL_PATH = "AdaBoost.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
else:
    model = None

# Custom Premium HTML layout with integrated CSS styling
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaBoost Model Deployment</title>
    <style>
        :root {
            --bg-color: #0f172a;
            --card-bg: #1e293b;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
            --accent: #0ea5e9;
            --accent-hover: #0284c7;
            --border: #334155;
        }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg-color);
            color: var(--text-main);
            margin: 0;
            padding: 40px 20px;
            display: flex;
            justify-content: center;
        }
        .container {
            width: 100%;
            max-width: 650px;
            background: var(--card-bg);
            padding: 35px;
            border-radius: 16px;
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3), 0 8px 10px -6px rgba(0, 0, 0, 0.3);
            border: 1px solid var(--border);
        }
        h2 {
            margin-top: 0;
            font-size: 28px;
            font-weight: 700;
            color: var(--text-main);
            border-bottom: 2px solid var(--border);
            padding-bottom: 15px;
        }
        p.subtitle {
            color: var(--text-muted);
            font-size: 14px;
            margin-top: -10px;
            margin-bottom: 30px;
        }
        .form-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        @media (max-width: 500px) {
            .form-grid { grid-template-columns: 1fr; }
        }
        .form-group {
            display: flex;
            flex-direction: column;
        }
        label {
            font-size: 14px;
            font-weight: 600;
            margin-bottom: 8px;
            color: var(--text-main);
        }
        input, select {
            background-color: var(--bg-color);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 10px 14px;
            color: var(--text-main);
            font-size: 15px;
            transition: border-color 0.2s;
        }
        input:focus, select:focus {
            outline: none;
            border-color: var(--accent);
        }
        .btn-container {
            grid-column: 1 / -1;
            margin-top: 15px;
        }
        button {
            width: 100%;
            background-color: var(--accent);
            color: #ffffff;
            border: none;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            border-radius: 8px;
            cursor: pointer;
            transition: background-color 0.2s;
        }
        button:hover {
            background-color: var(--accent-hover);
        }
        .result-box {
            margin-top: 30px;
            padding: 20px;
            border-radius: 8px;
            background-color: rgba(14, 165, 233, 0.15);
            border: 1px solid var(--accent);
            text-align: center;
        }
        .result-box h3 {
            margin: 0 0 5px 0;
            font-size: 16px;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        .result-val {
            font-size: 24px;
            font-weight: 700;
            color: #38bdf8;
        }
        .error-box {
            background-color: rgba(239, 68, 68, 0.15);
            border: 1px solid #ef4444;
            padding: 15px;
            border-radius: 8px;
            color: #fca5a5;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
</head>
<body>

<div class="container">
    <h2>Predictive Analytics Dashboard</h2>
    <p class="subtitle">Enter the metrics required by the AdaBoost classification engine below.</p>

    {% if error %}
    <div class="error-box">{{ error }}</div>
    {% endif %}

    <form method="POST" action="/predict">
        <div class="form-grid">
            <div class="form-group">
                <label for="age">Age</label>
                <input type="number" step="any" name="age" id="age" required value="{{ inputs.get('age', '') }}">
            </div>
            <div class="form-group">
                <label for="gender">Gender</label>
                <select name="gender" id="gender" required>
                    <option value="0" {% if inputs.get('gender') == '0' %}selected{% endif %}>Female</option>
                    <option value="1" {% if inputs.get('gender') == '1' %}selected{% endif %}>Male</option>
                </select>
            </div>
            <div class="form-group">
                <label for="tenure">Tenure (Months)</label>
                <input type="number" step="any" name="tenure" id="tenure" required value="{{ inputs.get('tenure', '') }}">
            </div>
            <div class="form-group">
                <label for="usage_frequency">Usage Frequency</label>
                <input type="number" step="any" name="usage_frequency" id="usage_frequency" required value="{{ inputs.get('usage_frequency', '') }}">
            </div>
            <div class="form-group">
                <label for="support_calls">Support Calls</label>
                <input type="number" step="any" name="support_calls" id="support_calls" required value="{{ inputs.get('support_calls', '') }}">
            </div>
            <div class="form-group">
                <label for="payment_delay">Payment Delay (Days)</label>
                <input type="number" step="any" name="payment_delay" id="payment_delay" required value="{{ inputs.get('payment_delay', '') }}">
            </div>
            <div class="form-group">
                <label for="subscription_type">Subscription Type</label>
                <select name="subscription_type" id="subscription_type" required>
                    <option value="0" {% if inputs.get('subscription_type') == '0' %}selected{% endif %}>Basic</option>
                    <option value="1" {% if inputs.get('subscription_type') == '1' %}selected{% endif %}>Standard</option>
                    <option value="2" {% if inputs.get('subscription_type') == '2' %}selected{% endif %}>Premium</option>
                </select>
            </div>
            <div class="form-group">
                <label for="contract_length">Contract Length</label>
                <select name="contract_length" id="contract_length" required>
                    <option value="0" {% if inputs.get('contract_length') == '0' %}selected{% endif %}>Monthly</option>
                    <option value="1" {% if inputs.get('contract_length') == '1' %}selected{% endif %}>Quarterly</option>
                    <option value="2" {% if inputs.get('contract_length') == '2' %}selected{% endif %}>Annual</option>
                </select>
            </div>
            <div class="form-group">
                <label for="total_spend">Total Spend</label>
                <input type="number" step="any" name="total_spend" id="total_spend" required value="{{ inputs.get('total_spend', '') }}">
            </div>
            <div class="form-group">
                <label for="last_interaction">Last Interaction (Days ago)</label>
                <input type="number" step="any" name="last_interaction" id="last_interaction" required value="{{ inputs.get('last_interaction', '') }}">
            </div>
            <div class="btn-container">
                <button type="submit">Run Model Inference</button>
            </div>
        </div>
    </form>

    {% if prediction is not none %}
    <div class="result-box">
        <h3>Inference Classification Result</h3>
        <div class="result-val">Class Designation: {{ prediction }}</div>
    </div>
    {% endif %}
</div>

</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Warning: 'AdaBoost.pkl' file was not discovered in the working directory.", inputs={})
    return render_template_string(HTML_TEMPLATE, inputs={})

@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return render_template_string(HTML_TEMPLATE, error="Error: Model deployment configuration missing.", inputs={})
    
    try:
        # Extract features matching the exact positions from your pickle header
        feature_values = [
            float(request.form.get("age")),
            float(request.form.get("gender")),
            float(request.form.get("tenure")),
            float(request.form.get("usage_frequency")),
            float(request.form.get("support_calls")),
            float(request.form.get("payment_delay")),
            float(request.form.get("subscription_type")),
            float(request.form.get("contract_length")),
            float(request.form.get("total_spend")),
            float(request.form.get("last_interaction"))
        ]
        
        # Reshape data array for single-sample inference
        input_data = np.array([feature_values])
        prediction = int(model.predict(input_data)[0])
        
        return render_template_string(HTML_TEMPLATE, prediction=prediction, inputs=request.form)
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error=f"Inference Failure: {str(e)}", inputs=request.form)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
