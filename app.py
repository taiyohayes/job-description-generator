# -*- coding: utf-8 -*-
"""ITP 459 Final Project - Amazon Careers.ipynb

# ITP 459 Final Project
**Group Members**: Rimi Bhardwaj, Taiyo Hayes, Tyler Kuang, Preston Doll

**Objective**: Create a platform that generates job descriptions similar to those previously posted on Amazon
Careers, maintaining the format and style of the original descriptions. The system should allow
users to input specific details to tailor the job descriptions to new roles.

# User Input Requirements

Users must provide a minimum of six new inputs via text
boxes: job title, years of experience, employment type (full-time or contract),
compensation range, work location (remote or in-person), and required qualifications.
Additionally, users should upload seven previous job descriptions from which the system
will derive the format and style.
"""

import streamlit as st
from scikit-learn.feature_extraction.text import CountVectorizer
from scikit-learn.metrics.pairwise import cosine_similarity
import pandas as pd
import os
from openai import OpenAI
from python-dotenv import load_dotenv

# LOGGER = st.get_logger(__name__)
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Handling and reformatting the uploaded files into a dictionary
def handle_files(files):
    job_descriptions_dict = {}
    for file in files:
        df = pd.read_csv(file)
        for i, row in df.iterrows():
            job_descriptions_dict[row["Title"]] = row["Description"]

    return job_descriptions_dict

# Function to interact with the OpenAI API to generate responses from the ChatGPT model based on a prompt
def get_completion(prompt, model="gpt-3.5-turbo"):
    # Create a list with one message dictionary, containing the role 'user' and the given prompt
    messages = [{"role": "user", "content": prompt}]

    # Call the OpenAI API to generate a chat completion using the specified model and messages
    # The temperature is set to 0 for deterministic output
    response = client.chat.completions.create(model=model,
                                            messages=messages,
                                            temperature=0.4)

    # Return the content of the first message in the response's choices
    return response.choices[0].message.content.strip()

# Job Description Matching
# The system will first identify the most suitable uploaded job description (using cosine similairty) based on the provided inputs.
def match_job_description(job_title, years_of_experience, employment_type, compensation_range, work_location, required_qualifications, job_descriptions):
    # Combine user inputs into a single string
    user_input = f"Job Title: {job_title}\nYears of Experience: {years_of_experience}\nEmployment Type: {employment_type}\nCompensation Range: {compensation_range}\nWork Location: {work_location}\nRequired Qualifications: {required_qualifications}"

    best_match = None
    best_similarity = 0

    # Vectorize user inputs for cosine similarity calculation
    vectorizer = CountVectorizer()
    user_vector = vectorizer.fit_transform([user_input])

    # Iterate through uploaded job descriptions
    for title, desc in job_descriptions.items():
        # Read the content of the uploaded file
        text = f"{title}\n{desc}"

        # Vectorize job description for cosine similarity calculation
        job_vector = vectorizer.transform([text])

        # Calculate cosine similarity between user inputs and job description
        similarity_score = cosine_similarity(user_vector, job_vector)[0][0]

        # Update best match if the current job description has a higher similarity score
        if similarity_score > best_similarity:
            best_similarity = similarity_score
            best_match = title

    return best_match

# Content Adaptation
# Modify the selected job description to reflect the new inputs and adjust the responsibilities as necessary to fit the new role. 
# Use GPT to create the modified job description.
def adapt_content(description, job_title, years_of_experience, employment_type, compensation_range, work_location, qualifications):
    prompt = f"""
        Please generate a new job description based on the provided information. Take the existing job description as a baseline, but create a new description based on the new job details (also provided below). Make sure all the new details are included somewhere in the new job description. Keep in mind that the company should remain unspecified unless it is provided in the new job details.\n\n 
        This is the existing job description: {description}\n\n
        These are the details of the new job:
        Job Title: {job_title}\n\tRequired Years of Experience: {years_of_experience}\n\tEmployment Type: {employment_type}\n\tCompensation Range: {compensation_range}\n\tWork Location: {work_location}\n\tRequired Qualifications: {qualifications}
    """
    adapted_description = get_completion(prompt)
    return adapted_description



st.title("AI Agent - Job Description Generator")
st.image('USC_top_image.png', use_column_width=True)

option = st.selectbox("Generate a new job description or proofread an existing one?",
                      ("Generate New Job Description", "Proofread an existing job description"))

if option == "Generate New Job Description":
    with st.form("job description generator"):
        st.subheader("Enter Job Details:")
        job_title = st.text_input("Job Title", key="title")
        years_of_experience = st.number_input("Years of Experience", min_value=0, step=1, key="experience")
        employment_type = st.selectbox("Employment Type", ["Full-time", "Contract"], key="employment")
        compensation_range = st.text_input("Compensation Range", key="compensation")
        work_location = st.selectbox("Work Location", ["Remote", "In-person"], key="location")
        required_qualifications = st.text_area("Required Qualifications", key="qualifications")
        uploaded_files = st.file_uploader("Upload job description examples (optional)", accept_multiple_files=True, type=['csv'])
        
        submit = st.form_submit_button("Generate Custom Job Description")

    if submit: #this button will trigger the generation of a custom job description
        # Get job names and descriptions based on the uploaded files
        job_descriptions = handle_files(uploaded_files)

        # Find best matching job description
        best_match_job = match_job_description(job_title, years_of_experience, employment_type, compensation_range, work_location, required_qualifications, job_descriptions)
        msg = "Matched job is: " + best_match_job
        # LOGGER.info(msg)

        # Adapt the matching job description to include new job specifications using GPT
        adapted_description = adapt_content(job_descriptions[best_match_job], job_title, years_of_experience, employment_type, compensation_range, work_location, required_qualifications)

        # Show newly generated custom job description to the user
        st.subheader("Customized Job Description:")
        st.write(adapted_description)

# EXTRA CREDIT: allow option for proofreading an existing 
else:
  st.subheader("Enter Proofreading Details:")
  user_description = st.text_area("Your Job Description")
  proofreading_level = st.selectbox("Select Proofreading Level",
                                    ("Minor", "Medium", "High"))
  response = get_completion(f"""Proofread the following job description at the designated proofreading level.
            If the proofreading level is listed as minor, simply correct for grammatical errors and absolutely do not change the structure
            or wording of the writing. If the level is high, you must completely modify sentences and the strcuture of the description, in an
            effort to have the best, most professional writing possible. If the level is medium, do something in between -- do not 
            completely restructure or rewrite entire sentences, but change some wording to sound more professional, and correct all grammatical errors.
            \n\nJob Description:{user_description}\n\nProofreading Level:{proofreading_level}
            """)
  if st.button("Proofread My Job Description"):
    st.write("Proofreading Results:")
    st.write(response)
