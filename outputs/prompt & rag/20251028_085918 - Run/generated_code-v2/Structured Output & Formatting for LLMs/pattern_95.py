import os
from typing import List, Literal

import openai
import instructor
from pydantic import BaseModel, Field

openai.api_key = os.getenv("OPENAI_API_KEY")
instructor.patch()

class Resource(BaseModel):
    type: Literal["article", "video", "book", "course"]
    title: str
    url: str

class Activity(BaseModel):
    name: str
    description: str

class AssessmentMethod(BaseModel):
    type: Literal["quiz", "assignment", "project"]
    details: str

class Lesson(BaseModel):
    lesson_title: str = Field(description="Title of the lesson")
    learning_objectives: List[str] = Field(description="Key objectives learners should achieve")
    recommended_resources: List[Resource] = Field(description="List of recommended resources for the lesson")
    activities: List[Activity] = Field(description="Hands-on activities or exercises")
    assessment_methods: List[AssessmentMethod] = Field(description="Methods to assess learning")
    estimated_duration_minutes: int = Field(description="Estimated duration of the lesson in minutes")

class LearningPath(BaseModel):
    path_title: str = Field(description="Overall title of the learning path")
    description: str = Field(description="A brief description of the learning path")
    lessons: List[Lesson] = Field(description="A list of individual lessons within the path")

def generate_personalized_learning_path(
    user_profile: str,
    topic: str,
    learning_style: str,
) -> LearningPath:
    prompt = f"""Generate a personalized learning path for a user with the following profile:
    User Profile: {user_profile}
    Learning Style: {learning_style}
    Topic: {topic}

    The learning path should be structured as a series of lessons, with each lesson including learning objectives, recommended resources, activities, assessment methods, and an estimated duration. Ensure the output is in JSON format conforming to the defined schema.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        response_model=LearningPath,
        messages=[
            {"role": "system", "content": "You are an AI assistant specialized in creating detailed and structured learning paths."},
            {"role": "user", "content": prompt},
        ],
    )
    return response

if __name__ == "__main__":
    # Example Usage
    user_profile_example = "Beginner in programming, interested in data science."
    learning_style_example = "Prefers hands-on exercises and video tutorials."
    topic_example = "Introduction to Python for Data Science"

    print("Generating personalized learning path...")
    try:
        learning_path = generate_personalized_learning_path(
            user_profile=user_profile_example,
            topic=topic_example,
            learning_style=learning_style_example,
        )
        print("Successfully generated learning path:")
        print(learning_path.json(indent=2))

        # You can then integrate this `learning_path` object with an LMS or for automated evaluation

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please ensure your OPENAI_API_KEY environment variable is set correctly and the model can be accessed.")
