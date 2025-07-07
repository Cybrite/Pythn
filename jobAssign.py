import pandas as pd

def load_job_data(filepath):
    df = pd.read_csv(filepath)
    job_data = {}
    for _, row in df.iterrows():
        role = row['Role']
        skills = {skill.strip().lower() for skill in row['Skills'].split(',')}
        job_data[role] = skills
    return job_data

def get_user_skills():
    print("Enter your skills one by one (type 'done' to finish):")
    user_skills = set()
    while True:
        skill = input("Skill: ").strip().lower()
        if skill == "done":
            break
        if skill:
            user_skills.add(skill)
    return user_skills


def recommend_job(user_skills, job_data):
    match_scores = {}
    for role, skills_required in job_data.items():
        match_count = len(user_skills & skills_required)
        match_scores[role] = match_count

    best_match = max(match_scores, key=match_scores.get)
    if match_scores[best_match] == 0:
        return None, 0
    return best_match, match_scores[best_match]

def main():
    file_path = "Employee_Survey_Analysis.csv"
    job_data = load_job_data(file_path)
    
    print("=== Job Recommendation System ===")
    user_skills = get_user_skills()
    
    if not user_skills:
        print("No skills entered.")
        return
    
    best_role, match_score = recommend_job(user_skills, job_data)
    if best_role:
        print(f"\n🎯 Recommended Job: {best_role}")
        print(f"✅ Skills matched: {match_score}")
    else:
        print("\n❌ No suitable job found for the provided skills.")

if __name__ == "__main__":
    main()
