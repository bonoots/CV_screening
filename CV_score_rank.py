import os
import re
import csv
import fitz  # PyMuPDF
import openai
import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text content from PDF file"""
    doc = fitz.open(file_path)
    text = "\n".join(page.get_text() for page in doc)
    return text

def extract_candidate_info(text: str) -> Dict[str, str]:
    info = {
        "name": "",
        "email": "",
        "phone": "",
        "university": "",
        "location": "",
    }
    name_match = re.findall(r"(?i)([A-Z][a-z]+\s[A-Z][a-z]+)", text)
    email_match = re.findall(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    phone_match = re.findall(r"\+?\d[\d\s\-()]{8,}\d", text)
    university_match = re.findall(r"(?i)([A-Z][\w\s]+University)", text)
    location_match = re.findall(r"(?i)(\b(?:New York|London|San Francisco|Toronto|Berlin|Paris|[A-Z][a-z]+)\b)", text)

    if name_match:
        info["name"] = name_match[0]
    if email_match:
        info["email"] = email_match[0]
    if phone_match:
        info["phone"] = phone_match[0]
    if university_match:
        info["university"] = university_match[0]
    if location_match:
        info["location"] = location_match[0]
    
    return info

def get_embedding(text: str) -> List[float]:
    """Get text embedding using OpenAI API"""
    try:
        text = text[:15000]  # Safe truncation
        response = openai_client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"OpenAI API error: {e}")
        return [0.0] * 1536  # Return zero vector on error

def score_resumes(resume_texts: List[str], jd_text: str) -> List[float]:
    """Calculate similarity scores between resumes and job description"""
    jd_vector = np.array(get_embedding(jd_text)).reshape(1, -1)
    scores = []
    
    for resume in resume_texts:
        res_vec = np.array(get_embedding(resume)).reshape(1, -1)
        sim = cosine_similarity(jd_vector, res_vec)[0][0]
        scores.append(sim)
        
    return scores

def process_all_resumes(resume_dir: str, jd_path: str, output_csv: str):
    """Main processing function"""
    # Verify input paths exist
    if not os.path.exists(resume_dir):
        raise FileNotFoundError(f"Resume directory not found: {resume_dir}")
    if not os.path.exists(jd_path):
        raise FileNotFoundError(f"Job description file not found: {jd_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    
    # Process resumes
    resumes = [f for f in os.listdir(resume_dir) if f.lower().endswith(".pdf")]
    if not resumes:
        raise ValueError("No PDF resumes found in directory")
    
    resume_texts = []
    candidates_info = []
    
    for file in resumes:
        path = os.path.join(resume_dir, file)
        try:
            text = extract_text_from_pdf(path)
            resume_texts.append(text)
            candidates_info.append(extract_candidate_info(text))
        except Exception as e:
            print(f"Error processing {file}: {e}")
            continue
    
    # Process job description
    jd_text = extract_text_from_pdf(jd_path)
    
    # Score resumes
    scores = score_resumes(resume_texts, jd_text)
    
    # Combine results
    for i, score in enumerate(scores):
        candidates_info[i]["score"] = round(score * 100, 2)  # Convert to percentage
    
    # Rank candidates
    ranked = sorted(candidates_info, key=lambda x: x["score"], reverse=True)
    for i, c in enumerate(ranked):
        c["rank"] = i + 1
    
    # Write output
    try:
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["rank", "name", "email", "phone", 
                                                 "location", "university", "score"])
            writer.writeheader()
            writer.writerows(ranked)
        print(f"Successfully saved results to: {output_csv}")
    except PermissionError:
        raise PermissionError(f"Cannot write to {output_csv}. Check directory permissions.")

# Example usage with proper paths
if __name__ == "__main__":
    resume_dir = "/Users/bonoots/Desktop/Personal/Jbujb/Recruitment/Software_eng/"
    jd_path = "/Users/bonoots/Desktop/Personal/Jbujb/Recruitment/JD/load_testing_monitoring_jd.pdf"
    output_csv = "/Users/bonoots/Desktop/Personal/Jbujb/Recruitment/ranked_candidates.csv"
    
    process_all_resumes(resume_dir, jd_path, output_csv)
