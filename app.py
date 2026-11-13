
import os
import gradio as gr
from dotenv import load_dotenv
from google import genai

# ==========================================
# 1. ENVIRONMENT CONFIGURATION
# ==========================================

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found in .env file. "
        "Please check your environment configuration."
    )

# Initialize Gemini client
client = genai.Client(api_key=api_key)


# ==========================================
# 2. GEMINI MODEL CONFIGURATION
# ==========================================

MODEL_NAME = "gemini-3.6-flash"


# ==========================================
# 3. AI LEARNING PATH GENERATION
# ==========================================


def generate_learning_path(
    name,
    target_role,
    current_skills,
    experience,
    weekly_hours,
    duration,
    learning_style
):

    if not name or not target_role or not current_skills:
        return "Please complete your name, target career and current skills."

    if weekly_hours is None or not 1 <= weekly_hours <= 80:
        return "Please enter study hours between 1 and 80."

    prompt = f"""
You are an expert career learning path advisor.

Create a practical personalized learning roadmap.

LEARNER:
Name: {name}
Career Goal: {target_role}
Skills: {current_skills}
Experience: {experience}
Weekly Hours: {weekly_hours}
Duration: {duration}
Learning Style: {learning_style}

Generate these 11 sections:

1. Learner Profile Analysis
2. Existing Skills Analysis
3. Missing Skills and Skill Gap Analysis
4. Personalized Learning Roadmap
5. Weekly Study Schedule
6. Recommended Learning Topics
7. Hands-on Practice Exercises
8. Mini Projects
9. Recommended Learning Resources
10. Weekly Milestones
11. Final Assessment

INSTRUCTIONS:
- Personalize every section.
- Prioritize essential career skills.
- Respect the available study hours.
- Organize topics in learning order.
- Include practical exercises and projects.
- Provide measurable weekly milestones.
- Keep the roadmap realistic.
- Use Markdown headings, tables and bullet points.
- Keep explanations concise but useful.
- Do not invent resource URLs.
- Mention if the selected duration is insufficient
  for full career readiness.

Generate a structured, actionable roadmap.
"""

    try:

        print("Requesting Gemini 3.8 Flash...", flush=True)

        response = client.models.generate_content(
            model="gemini-3.8-flash",
            contents=prompt,
            config={
                "max_output_tokens": 3000,
                "http_options": {
                    "timeout": 180000
                }
            }
        )

        if response.text:
            print("Roadmap generated successfully!", flush=True)
            return response.text

        return "Gemini returned an empty response. Please try again."

    except Exception as e:

        print(f"Gemini 3.8 Error: {e}", flush=True)

        return (
            "## Gemini 3.8 Generation Failed\n\n"
            f"**Error:** `{str(e)}`\n\n"
            "The model may be experiencing high demand or "
            "a server timeout. Please try again later."
        )



# ==========================================
# 6. GRADIO APPLICATION INTERFACE
# ==========================================

with gr.Blocks(
    title="AI Personalized Learning Path Generator"
) as app:

    gr.Markdown(
        """
        # 🎓 AI Personalized Learning Path Generator

        ### Build your personalized learning journey with Generative AI.

        Enter your learner profile and let Gemini analyze
        your skills and create a personalized learning roadmap.
        """
    )

    gr.Markdown("## 👤 Learner Profile")

    with gr.Row():

        name = gr.Textbox(
            label="Your Name",
            placeholder="Enter your name"
        )

        target_role = gr.Dropdown(
            choices=[
                "Data Scientist",
                "Data Analyst",
                "AI Engineer",
                "Machine Learning Engineer",
                "Python Developer",
                "Software Developer",
                "Other"
            ],
            label="Target Career / Goal"
        )

    current_skills = gr.Textbox(
        label="Current Skills",
        placeholder="Example: Python, SQL, Statistics, HTML",
        lines=3
    )

    with gr.Row():

        experience = gr.Dropdown(
            choices=[
                "Beginner",
                "Intermediate",
                "Advanced"
            ],
            label="Experience Level"
        )

        weekly_hours = gr.Number(
            label="Weekly Study Hours",
            value=10,
            minimum=1,
            maximum=80,
            precision=0
        )

    with gr.Row():

        duration = gr.Dropdown(
            choices=[
                "2 weeks",
                "4 weeks",
                "6 weeks",
                "8 weeks",
                "12 weeks",
                "6 months"
            ],
            label="Target Completion Period"
        )

        learning_style = gr.Dropdown(
            choices=[
                "Theory-focused",
                "Practical-focused",
                "Balanced learning"
            ],
            label="Preferred Learning Style"
        )

    gr.Markdown("---")

    generate_btn = gr.Button(
        "🚀 Generate My Learning Roadmap",
        variant="primary"
    )

    gr.Markdown(
        "## 📚 Your Personalized Learning Roadmap"
    )

    output = gr.Markdown(
        "Your generated learning roadmap will appear here."
    )

    # Connect button to Gemini generation function
    generate_btn.click(
        fn=generate_learning_path,
        inputs=[
            name,
            target_role,
            current_skills,
            experience,
            weekly_hours,
            duration,
            learning_style
        ],
        outputs=output,
        show_progress="minimal"
    )


# ==========================================
# 7. APPLICATION LAUNCH
# ==========================================

if __name__ == "__main__":

    app.launch(
        share=True,
        theme=gr.themes.Soft()
    )
