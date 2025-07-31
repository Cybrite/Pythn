import csv
import os

def determine_job_position(travel_preference):
    """
    Determine job position based on travel preferences.
    
    Parameters:
    - travel_preference: String representing travel preference
    
    Returns:
    - String representing the job position
    """
    if travel_preference == 'No':
        return 'At the Main Shops'
    elif travel_preference == 'Yes (Only within town)':
        return 'Inside Rourkela (At customer\'s place)'
    elif travel_preference == 'Yes (Anywhere)':
        return 'Outside Rourkela (At customer\'s place)'
    else:
        return 'Unknown position'

def process_employee_data(csv_file):
    """
    Process employee data from CSV file and determine potential position changes.
    
    Parameters:
    - csv_file: Path to the CSV file containing employee data
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip the first two rows (headers)
            next(csv_reader)
            next(csv_reader)
            
            print("\n" + "-" * 100)
            print(f"{'Employee Name':<25} {'Current Position':<25} {'Recommended Position':<25} {'Status':<25}")
            print("-" * 100)
            
            changed_count = 0
            no_change_count = 0
            not_willing_count = 0
            
            for row in csv_reader:
                if len(row) < 11:  # Ensure the row has enough elements
                    continue
                
                name = row[1].strip()
                current_position = row[7].strip()  # Current Job Posting
                travel_preference = row[9].strip()  # Travel preference
                willing_to_shift = row[11].strip()  # Willingness to shift
                
                recommended_position = determine_job_position(travel_preference)
                
                if willing_to_shift == 'Yes' and current_position != recommended_position:
                    status = "Position Change Recommended"
                    changed_count += 1
                elif willing_to_shift != 'Yes':
                    status = "Not Willing to Shift"
                    not_willing_count += 1
                else:
                    status = "No Change Needed"
                    no_change_count += 1
                
                print(f"{name:<25} {current_position:<25} {recommended_position:<25} {status:<25}")
            
            print("-" * 100)
            print(f"\nSummary:")
            print(f"Total employees processed: {changed_count + no_change_count + not_willing_count}")
            print(f"Employees recommended for position change: {changed_count}")
            print(f"Employees with no position change needed: {no_change_count}")
            print(f"Employees not willing to shift: {not_willing_count}")
            
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def generate_position_change_report(csv_file, output_file="position_change_report.csv"):
    """
    Generate a detailed report of recommended position changes.
    
    Parameters:
    - csv_file: Path to the CSV file containing employee data
    - output_file: Path to save the output report
    """
    try:
        with open(csv_file, 'r', encoding='utf-8') as file:
            csv_reader = csv.reader(file)
            
            # Skip the first two rows (headers)
            next(csv_reader)
            next(csv_reader)
            
            # Prepare output data
            output_data = [["Employee Name", "Current Role", "Current Position", "Recommended Position", "Travel Preference", "Willing to Shift", "Status"]]
            
            for row in csv_reader:
                if len(row) < 11:  # Ensure the row has enough elements
                    continue
                
                name = row[1].strip()
                role = row[2].strip()
                current_position = row[7].strip()
                travel_preference = row[9].strip()
                willing_to_shift = row[11].strip()
                
                recommended_position = determine_job_position(travel_preference)
                
                if willing_to_shift == 'Yes' and current_position != recommended_position:
                    status = "Position Change Recommended"
                elif willing_to_shift != 'Yes':
                    status = "Not Willing to Shift"
                else:
                    status = "No Change Needed"
                
                output_data.append([name, role, current_position, recommended_position, travel_preference, willing_to_shift, status])
            
            # Write to output file
            with open(output_file, 'w', newline='', encoding='utf-8') as outfile:
                csv_writer = csv.writer(outfile)
                csv_writer.writerows(output_data)
            
            print(f"\nDetailed report saved to {output_file}")
            
    except Exception as e:
        print(f"An error occurred generating the report: {str(e)}")

# Main execution
if __name__ == "__main__":
    csv_file = r"d:\Pythn\Employee Survey Form (Responses) - Form responses 1.csv"
    
    print("\nProcessing employee data for position assignments...")
    process_employee_data(csv_file)
    
    generate_position_change_report(csv_file)
    print("\nProcess completed!")
