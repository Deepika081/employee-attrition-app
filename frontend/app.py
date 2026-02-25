import gradio as gr
import requests

# Function to call FastAPI
def predict_attrition_gradio(
    satisfaction_level, 
    last_evaluation, 
    number_project, 
    average_montly_hours, 
    time_spend_company, 
    Work_accident, 
    promotion_last_5years, 
    Departments, 
    salary
):
    payload = {
        "satisfaction_level": satisfaction_level,
        "last_evaluation": last_evaluation,
        "number_project": number_project,
        "average_montly_hours": average_montly_hours,
        "time_spend_company": time_spend_company,
        "Work_accident": Work_accident,
        "promotion_last_5years": promotion_last_5years,
        "Departments": Departments,
        "salary": salary
    }

    try:
        response = requests.post("http://backend:8000/predict", json=payload)
        data = response.json()
        
        leave_pct = round(data['leave_probability'] * 100)
        
        output_lines = [
            f"🔹 Prediction: {data['prediction']}",
            f"🔹 Leave Probability: {leave_pct}%",
            f"🔹 Risk Level: {data['risk_level']}",
            "",
            "Top 3 Factors Influencing Prediction:"
        ]
        for factor in data["top_factors"]:
            # ↑ means contributing to leave, ↓ means reducing leave risk
            impact_symbol = "↑" if "increase" in factor["impact"] else "↓"
            output_lines.append(f"  {impact_symbol} {factor['feature']} ({factor['contribution_strength']})")

        return "\n".join(output_lines)

    except Exception as e:
        return f"Error connecting to backend: {e}"

# Gradio interface
iface = gr.Interface(
    fn=predict_attrition_gradio,
    inputs=[
        gr.Slider(0, 1, step=0.01, label="Satisfaction Level"),
        gr.Slider(0, 1, step=0.01, label="Last Evaluation"),
        gr.Number(label="Number of Projects"),
        gr.Number(label="Average Monthly Hours"),
        gr.Number(label="Time Spent in Company (Years)"),
        gr.Radio([0,1], label="Work Accident?"),
        gr.Radio([0,1], label="Promotion in last 5 years?"),
        gr.Dropdown(['sales','accounting','hr','technical','support','management','IT','product_mng','marketing','RandD'], label="Department"),
        gr.Dropdown(['low','medium','high'], label="Salary Level")
    ],
    outputs=gr.Textbox(label="Attrition Prediction & Factors"),
    title="Employee Attrition Predictor",
    description="Predict if an employee is likely to leave and see top contributing factors."
)

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)