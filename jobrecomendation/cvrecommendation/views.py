from django.shortcuts import render
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from rest_framework.decorators import api_view
from rest_framework.response import Response

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

def encode_text(text_list):
    embeddings = model.encode(text_list, convert_to_tensor=True)
    return embeddings

def calculate_similarity(job_desc, resumes):
    texts = [job_desc] + resumes
    embeddings = encode_text(texts)
    job_desc_embedding = embeddings[0]
    resume_embeddings = embeddings[1:]
    similarity_scores = cosine_similarity(job_desc_embedding.unsqueeze(0), resume_embeddings)
    return similarity_scores.flatten()

def selected_resume(job_desc, resumes, top_n=5, threshold=0.5):
    similarity_scores = calculate_similarity(job_desc, resumes)
    ranked_indices = np.argsort(similarity_scores)[::-1]
    top_resumes = []
    for i in ranked_indices:
        score = similarity_scores[i]
        resume = resumes[i]
        status = "Candidate is the Best Fit" if score >= threshold else "Candidate is a Potential Fit"
        top_resumes.append({"resume": resume, "score": float(score), "status": status})
    related = [r for r in top_resumes if r['status'] == "Candidate is the Best Fit"]
    not_related = [r for r in top_resumes if r['status'] == "Candidate is a Potential Fit"]
    grouped_resumes = related[:top_n] + not_related[:max(0, top_n - len(related))]
    return grouped_resumes

@api_view(['POST'])
def selected_resume_api(request):
    data = request.data
    job_desc = data.get('job_desc')
    resumes = data.get('resumes')

    if not job_desc:
        return Response({"error": "'job_desc' is required."}, status=400)
    if not isinstance(resumes, list) or len(resumes) == 0:
        return Response({"error": "'resumes' must be a non-empty list."}, status=400)

    try:
        top_n = int(data.get('top_n', 5))
        threshold = float(data.get('threshold', 0.5))

        if top_n <= 0:
            return Response({"error": "'top_n' must be a positive integer."}, status=400)
        if not (0.0 <= threshold <= 1.0):
            return Response({"error": "'threshold' must be between 0.0 and 1.0."}, status=400)

        concatenated_resumes = [
            f"{resume.get('bio', '')}. Experience: {resume.get('experience', '')}. Skills: {resume.get('skills', '')}. Specialization: {resume.get('specialization', '')}" 
            for resume in resumes
        ]

        ranked_resumes = selected_resume(job_desc, concatenated_resumes, top_n, threshold)

        detailed_responses = []
        for ranked_resume in ranked_resumes:
            for resume in resumes:
                concatenated_resume = f"{resume.get('bio', '')}. Experience: {resume.get('experience', '')}. Skills: {resume.get('skills', '')}. Specialization: {resume.get('specialization', '')}"
                if ranked_resume['resume'] == concatenated_resume:
                    detailed_responses.append({
                        "name": resume.get("name"),
                        "email": resume.get("email"),
                        "bio": resume.get("bio"),
                        "experience": resume.get("experience"),
                        "skills": resume.get("skills"),
                        "specialization": resume.get("specialization"),
                        "score": ranked_resume["score"],
                        "status": ranked_resume["status"]
                    })
                    break

        return Response({"ranked_resumes": detailed_responses}, status=200)
    except Exception as e:
        return Response({"error": f"An error occurred: {str(e)}"}, status=500)
