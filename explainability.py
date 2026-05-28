import shap
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Sample text
sample_text = [
    "Breaking news about politics and economy"
]

# Transform text
sample_vector = vectorizer.transform(sample_text)

# Create SHAP explainer
explainer = shap.LinearExplainer(model, sample_vector)

# Generate SHAP values
shap_values = explainer.shap_values(sample_vector)

print(shap_values)