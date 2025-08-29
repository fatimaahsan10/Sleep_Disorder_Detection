import gradio as gr
import joblib
import pandas as pd

# Load model
model = joblib.load("sleep_disorder_pipeline.joblib")

# Label mapping (change according to your dataset!)
label_map = {
    0: "No Sleep Disorder",
    1: "Sleep Disorder"
}

# Prediction function
def predict_sleep_disorder(age, sleep_duration, quality_of_sleep, physical_activity, stress_level, heart_rate, daily_steps, 
                           gender, occupation, bmi_category, bp_sys, bp_dia):
    try:
        input_data = pd.DataFrame([{
            "Age": age,
            "Sleep Duration": sleep_duration,
            "Quality of Sleep": quality_of_sleep,
            "Physical Activity Level": physical_activity,
            "Stress Level": stress_level,
            "Heart Rate": heart_rate,
            "Daily Steps": daily_steps,
            "Gender": gender,
            "Occupation": occupation,
            "BMI Category": bmi_category,
            "BP_sys": bp_sys,
            "BP_dia": bp_dia
        }])

        # Get prediction and probability
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        confidence = max(probs) * 100

        return f"✅ Predicted: {label_map[prediction]} (Confidence: {confidence:.2f}%)"

    except Exception as e:
        return f"⚠️ Error: {str(e)}"

# Inputs
inputs = [
    gr.Number(label="Age"),
    gr.Number(label="Sleep Duration (hours)"),
    gr.Number(label="Quality of Sleep (1-10)"),
    gr.Number(label="Physical Activity (minutes/day)"),
    gr.Number(label="Stress Level (1-10)"),
    gr.Number(label="Heart Rate (bpm)"),
    gr.Number(label="Daily Steps"),
    gr.Radio(["Male", "Female"], label="Gender"),
    gr.Textbox(label="Occupation"),
    gr.Dropdown(["Underweight", "Normal", "Overweight", "Obese"], label="BMI Category"),
    gr.Number(label="BP Systolic"),
    gr.Number(label="BP Diastolic")
]

# Output
output = gr.Textbox(label="Prediction")

# Example inputs for easy testing
examples = [
    [30, 6, 5, 40, 7, 75, 8000, "Male", "Engineer", "Normal", 120, 80],
    [45, 4, 3, 20, 9, 90, 4000, "Female", "Doctor", "Overweight", 140, 95],
    [22, 8, 9, 60, 3, 70, 10000, "Male", "Student", "Underweight", 110, 70]
]

# Gradio app
app = gr.Interface(
    fn=predict_sleep_disorder,
    inputs=inputs,
    outputs=output,
    title="Sleep Disorder Prediction App",
    description="Enter health & lifestyle details to predict possible sleep disorder.",
    examples=examples
)

if __name__ == "__main__":
    app.launch()
