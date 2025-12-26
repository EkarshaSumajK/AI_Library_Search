import csv
import random
import faker
from datetime import datetime

# Initialize Faker
fake = faker.Faker()

# Constants
NUM_BOOKS = 400
FILENAME = 'simulated_books.csv'

# Colleges and Majors for Context and Keywords
COLLEGE_CONTEXTS = {
    "College of Engineering": [
        "Civil Engineering", "Mechanical Engineering", "Electrical Engineering", 
        "Computer Engineering", "Chemical Engineering", "Architectural Engineering"
    ],
    "College of Medicine and Health Sciences": [
        "Internal Medicine", "Surgery", "Pediatrics", "Public Health", "Nursing", 
        "Pathology", "Anesthesiology"
    ],
    "College of Pharmacy": [
        "Clinical Pharmacy", "Pharmacology", "Pharmaceutics", "Medicinal Chemistry"
    ],
    "International Maritime College Oman": [
        "Marine Engineering", "Nautical Studies", "Port Management", 
        "Logistics & Supply Chain", "Maritime Law"
    ]
}

KEYWORDS_BY_DOMAIN = {
    "Engineering": ["thermodynamics", "structural analysis", "circuits", "robotics", "fluid mechanics", "materials science", "sustainable design", "microprocessors"],
    "Medicine": ["anatomy", "physiology", "immunology", "epidemiology", "diagnostic imaging", "patient care", "medical ethics", "surgery techniques"],
    "Pharmacy": ["drug interaction", "pharmacokinetics", "biopharmaceutics", "toxicology", "natural products", "pharmaceutical care"],
    "Maritime": ["navigation", "marine safety", "shipping logistics", "maritime law", "oceanography", "port operations", "marine diesel engines"]
}

PUBLISHERS = [
    "Springer", "Wiley", "Elsevier", "McGraw-Hill Education", "Pearson", 
    "SAGE Publications", "Cambridge University Press", "Oxford University Press", 
    "Routledge", "Taylor & Francis"
]

FORMATS = ["Hardcover", "Paperback", "E-Book"]
LANGUAGES = ["English"] * 95 + ["Arabic"] * 5  # Mostly English

def generate_book_data():
    books = []
    
    print(f"Generating {NUM_BOOKS} simulated books...")
    
    for i in range(NUM_BOOKS):
        # Pick a domain/college
        college = random.choice(list(COLLEGE_CONTEXTS.keys()))
        major = random.choice(COLLEGE_CONTEXTS[college])
        
        # Determine broad domain for keywords
        if "Engineering" in college:
            domain = "Engineering"
        elif "Medicine" in college:
            domain = "Medicine"
        elif "Pharmacy" in college:
            domain = "Pharmacy"
        else:
            domain = "Maritime"
            
        # Generate specific attributes
        title_prefix = fake.bs().title()
        subject_keyword = random.choice(KEYWORDS_BY_DOMAIN[domain])
        title = f"{title_prefix}: {subject_keyword.capitalize()} and {fake.word().capitalize()}"
        if random.random() < 0.3:
            title = f"Fundamentals of {subject_keyword.capitalize()}"
        elif random.random() < 0.3:
            title = f"{subject_keyword.capitalize()} in {major}"

        author = fake.name()
        if random.random() < 0.4:
            author += f", {fake.name()}"
            
        # Description
        description = fake.paragraph(nb_sentences=5)
        description += f" This comprehensive text explores {subject_keyword} within the context of {major}. "
        description += "Key topics include analysis, design methodologies, and practical applications. "
        description += "Ideal for students and professionals seeking depth in the field."
        
        # Link
        biblionumber = random.randint(10000, 99999)
        link = f"https://elibrary.nu.edu.om/cgi-bin/koha/opac-detail.pl?biblionumber={biblionumber}"
        
        # Rating
        rating = round(random.uniform(3.5, 5.0), 1)
        
        # Keywords
        keywords = f"{subject_keyword}, {major.lower()}, {domain.lower()}, {fake.word()}, research, textbook"
        
        # Context (simulating the 'Context' column in new_data.csv)
        context = f"Designed for students and researchers in {major}, focusing on {subject_keyword}."
        
        book = {
            'Title': title,
            'Author(s)': author,
            'Description': description,
            'Publisher': random.choice(PUBLISHERS),
            'Link': link,
            'Rating': rating,
            'Keywords': keywords,
            'ISBN': fake.isbn13(),
            'Publication Year': random.randint(1980, 2024),
            'Pages': random.randint(150, 900),
            'Language': random.choice(LANGUAGES),
            'Format': random.choice(FORMATS),
            'Context': context,
            'College': college  # Explicitly add college
        }
        books.append(book)
        
    return books

def save_to_csv(books):
    fieldnames = [
        'Title', 'Author(s)', 'Description', 'Publisher', 'Link', 'Rating', 
        'Keywords', 'ISBN', 'Publication Year', 'Pages', 'Language', 'Format', 'Context', 'College'
    ]
    
    with open(FILENAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(books)
        
    print(f"Successfully saved {len(books)} books to {FILENAME}")

if __name__ == "__main__":
    books = generate_book_data()
    save_to_csv(books)
