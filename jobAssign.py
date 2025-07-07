import pandas as pd

'''This is a job assignment system that recommends job roles based on employee skills.
It can handle both individual job recommendations and batch assignments from a CSV file.
Columns in the CSV file should include columns strictly named as:
- Employee Full Name
- Skills '''

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

def load_employee_data(filepath):
    """Load employee data from CSV file with 'Employee Full Name' and 'Skills' columns"""
    df = pd.read_csv(filepath)
    employee_data = {}
    for _, row in df.iterrows():
        name = row['Employee Full Name']
        skills = {skill.strip().lower() for skill in row['Skills'].split(',')}
        employee_data[name] = skills
    return employee_data

def assign_jobs_to_employees(employee_filepath, job_filepath):
    """Assign jobs to multiple employees from CSV file"""
    print("=== Batch Job Assignment System ===")
    
    job_data = load_job_data(job_filepath)
    employee_data = load_employee_data(employee_filepath)
    
    assignments = []
    
    print(f"\nProcessing {len(employee_data)} employees...\n")
    
    for employee_name, employee_skills in employee_data.items():
        best_role, match_score = recommend_job(employee_skills, job_data)
        
        assignment = {
            'Employee Name': employee_name,
            'Skills': ', '.join(sorted(employee_skills)),
            'Recommended Role': best_role if best_role else 'No suitable role found',
            'Skills Matched': match_score
        }
        assignments.append(assignment)
        
        if best_role:
            print(f"👤 {employee_name}")
            print(f"   🎯 Assigned Role: {best_role}")
            print(f"   ✅ Skills Matched: {match_score}")
            print(f"   🔧 Skills: {', '.join(sorted(employee_skills))}")
        else:
            print(f"👤 {employee_name}")
            print(f"   ❌ No suitable role found")
            print(f"   🔧 Skills: {', '.join(sorted(employee_skills))}")
        print("-" * 50)
    
    output_df = pd.DataFrame(assignments)
    output_filename = "job_assignments.csv"
    output_df.to_csv(output_filename, index=False)
    
    print(f"\n📊 Assignment summary:")
    print(f"   Total employees processed: {len(assignments)}")
    assigned_count = sum(1 for a in assignments if a['Recommended Role'] != 'No suitable role found')
    print(f"   Successfully assigned: {assigned_count}")
    print(f"   Unassigned: {len(assignments) - assigned_count}")
    print(f"   Results saved to: {output_filename}")
    
    return assignments

def main():
    print("Choose an option:")
    print("1. Individual job recommendation")
    print("2. Batch job assignment from CSV")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        # Original individual recommendation
        file_path = "Employee_Survey_Analysis.csv"
        job_data = load_job_data(file_path)
        
        print("\n=== Job Recommendation System ===")
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
    
    elif choice == "2":
        employee_file = input("Enter the path to the employee CSV file: ").strip()
        job_file = "Employee_Survey_Analysis.csv"
        
        try:
            assign_jobs_to_employees(employee_file, job_file)
        except FileNotFoundError:
            print(f"Error: Could not find the file '{employee_file}'")
        except KeyError as e:
            print(f"Error: Missing required column {e} in the CSV file")
            print("Make sure your CSV has 'Employee Full Name' and 'Skills' columns")
        except Exception as e:
            print(f"Error: {e}")
    
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
