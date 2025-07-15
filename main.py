# def main():
#     print("Hello from smart-student-agent!")


# if __name__ == "__main__":
#     main()

# import os
# from dotenv import load_dotenv
# from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel ,RunConfig
# # from uvicorn import Config

# load_dotenv()

# gemini_api_key = os.getenv("GEMINI_API_KEY")

# external_client= AsyncOpenAI(
#     api_key=gemini_api_key,
#     base_url="https://generativelanguage.googleapis.com/v1beta/openai",
# )
# # model = OpenAIChatCompletionsModel(
# #    model="gpt-4o",
# #      openai_client=external_client
# #      )

# model = OpenAIChatCompletionsModel(
#     model="gemini-2.0-flash",
#     openai_client=external_client,
# )
# config = RunConfig(
#      model=model,
#      model_provider=external_client,
#      tracing_disabled=True
#  )

# # from agents import Agent, Runner

# math_agent = Agent(name="Math Tutor", instructions="Help with math homework step by step.")
# history_agent = Agent(name="History Tutor", instructions="Explain historical topics clearly.")
# triage = Agent(name="Triage", instructions="Route question to the right tutor", handoffs=[math_agent, history_agent])

# result = Runner.run_sync(triage, "Why did the French Revolution happen?")
# print(result.final_output)

import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load your Gemini API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create a Gemini model
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

# --- Sub-agent functions ---
def math_tutor(question: str) -> str:
    prompt = f"You are a math tutor. Help step by step.\nQuestion: {question}"
    response = gemini_model.generate_content(prompt)
    return response.text

def history_tutor(question: str) -> str:
    prompt = f"You are a history tutor. Explain in simple terms.\nQuestion: {question}"
    response = gemini_model.generate_content(prompt)
    return response.text

def triage_agent(user_question: str) -> str:
    """
    Routes the question to the correct tutor based on subject keywords.
    """
    # Simple keyword-based routing
    math_keywords = ["math", "algebra", "geometry", "equation", "solve", "number"]
    history_keywords = ["history", "revolution", "war", "king", "ancient", "past"]

    lower_q = user_question.lower()
    if any(k in lower_q for k in math_keywords):
        return math_tutor(user_question)
    elif any(k in lower_q for k in history_keywords):
        return history_tutor(user_question)
    else:
        # Default to general explanation
        response = gemini_model.generate_content(
            f"You are a helpful student assistant. Answer clearly.\nQuestion: {user_question}"
        )
        return response.text


# --- Test the Smart Student Agent ---
question1 = "Why did the French Revolution happen?"
answer1 = triage_agent(question1)
print("\nQ:", question1)
print("A:", answer1)

question2 = "Solve 2x^2 + 3x + 1 = 0"
answer2 = triage_agent(question2)
print("\nQ:", question2)
print("A:", answer2)

