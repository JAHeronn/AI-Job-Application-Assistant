import os
import gradio as gr
import pdfplumber
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv

# Load environment variables (API key, etc.)
load_dotenv(override=True)

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("Missing OPENAI_API_KEY in environment variables.")

openai = OpenAI()
MODEL_GPT = "gpt-4o-mini"

def extract_file_text(file_path):
    """ Extracts text from a PDF file """
    try:
        with pdfplumber.open(file_path) as file:
            text = ""
            for page in file.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Error reading PDF file: {e}")

def stream_gpt(job_prompt, cv_file):
    """ Streams GPT output comparing job post and CV """
    try:
        if not job_prompt or not cv_file:
            yield "Please provide both a job description and a PDF CV."
            return

        cv_text = extract_file_text(cv_file)

        system_message = (
            "You are a helpful assistant who is especially tailored to assist with job applications. "
            "In particular, when given details about a job post and provided an accompanying CV, your role is to: "
            "1) Analyse all the description, requirements, and details of the job post "
            "2) Analyse all the details of the accompanying CV against the details and requirements of the job post "
            "3) Generate tailored CV bullet points, emphasising the relevant experience in the CV that matches the details of the job post "
            "4) Identify any skill gaps, whereby the job post has required or relevant skills beyond those which are in the CV, and provide "
            "learning recommendations/advice on how to bridge this gap to better meet the requirements of the job post "
            "5) Suggest cover letter talking points which are relevant to the details in the CV in relation to the job post "
            "6) Estimate a salary range based on the location, requirements and position of the given job, to highlight prospects of career progression if "
            "successfully hired in the given role."
        )

        user_prompt = f"""Here is an extract of a job post: {job_prompt}. And here is my CV: {cv_text}.
            Please write a detailed analysis in markdown, including:
            - A comparison of my CV against the job post, highlighting the details/experience in my CV that are relevant to the job post.
            - Identification of any existing skill gaps where the job post requires skills beyond those detailed in my CV.
            - Learning advice and recommendations on how to bridge any identified skill gaps.
            - Suggested relevant talking points for a cover letter, taking into account the details in my CV and the job post.
            - A salary range estimation for if I were hired in the given job role and progressed in my career."""

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]

        try:
            stream = openai.chat.completions.create(
                model=MODEL_GPT,
                messages=messages,
                stream=True
            )
        except OpenAIError as e:
            yield f"OpenAI API Error: {e}"
            return

        result = ""
        for chunk in stream:
            content = chunk.choices[0].delta.content
            if content:
                result += content
                yield result

    except Exception as e:
        yield f"Unexpected error: {str(e)}"

# Gradio interface
interface = gr.Interface(
    fn=stream_gpt,
    inputs=[
        gr.Textbox(
            label="Job Details",
            lines=10,
            placeholder="Paste the job description here..."
        ),
        gr.File(
            label="Upload Your CV (PDF only)",
            type="filepath",
            file_types=[".pdf"]
        )
    ],
    outputs=gr.Markdown(label="Response:"),
    title="AI Job Application Assistant",
    description="Upload your CV and paste a job description to get tailored application advice.",
    flagging_mode="never"
)


if __name__ == "__main__":
    try:
        interface.launch()
    except Exception as e:
        print(f"Failed to launch the app: {e}")
