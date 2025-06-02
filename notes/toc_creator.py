from bs4 import BeautifulSoup

# Path to your HTML file
CONST_html_file_path = './aws-ans/notes/hybrid-networks.html'
CONST_html_file_path = 'C:/codes/github-pages/christoferson.github.io/notes/aws-ans/notes/hybrid-networks.html'
CONST_html_file_path = 'C:/codes/github-pages/christoferson.github.io/notes/aws-ans/notes/hybrid-networks-dns.html'

# Function to extract the required elements from the HTML
def extract_toc(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Extract p with specific inline styles for blueviolet color and font size 20px (sections)
    p_elements = soup.find_all('p', style='color: blueviolet; font-size: 20px;')
    
    # Extract p with specific inline styles for #0066cc color and font size 16px (subsections)
    p_subsections = soup.find_all('p', style='color: #0066cc; font-size: 16px;')
    
    toc = []
    current_section = None  # Variable to track the current section

    # Loop through p elements with blueviolet color (main section headers)
    for p in p_elements:
        strong_tag = p.find('strong')
        if strong_tag:
            section_title = strong_tag.get_text(strip=True)
            current_section = {"type": "section", "title": section_title, "subsections": []}
            toc.append(current_section)  # Add this section to the TOC list
    
    # Loop through p elements with #0066cc color (subsection headers)
    for p in p_subsections:
        strong_tag = p.find('strong')
        if strong_tag and current_section:
            subsection_title = strong_tag.get_text(strip=True)
            current_section["subsections"].append({"type": "subsection", "title": subsection_title})
    
    return toc

# Function to create HTML for the table of contents
def generate_html_toc(toc):
    toc_html = """
    <div class="container mt-5">
        <p>Table of contents</p>
        <ul>
    """
    
    # Loop over the TOC and add the sections and subsections to the HTML structure
    for item in toc:
        if item["type"] == "section":
            toc_html += f'<li style="color: blueviolet; font-size: 16px; list-style-type: none;">{item["title"]}'
            toc_html += '<ul>'
            # Add any subsections if they exist
            for subsection in item["subsections"]:
                toc_html += f'<li style="color: #0066cc; font-size: 12px; list-style-type: none;">{subsection["title"]}</li>'
            toc_html += '</ul></li>'
    
    toc_html += """
        </ul>
    </div>
    """
    
    return toc_html

# Read the HTML file
def read_html(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Path to your HTML file
html_file_path = CONST_html_file_path

# Read HTML file content
html_content = read_html(html_file_path)

# Extract table of contents
toc = extract_toc(html_content)

# Print the results for debugging purposes
print("Table of Contents:")
for item in toc:
    print(item["title"])

# Generate HTML for the table of contents
toc_html = generate_html_toc(toc)

# Print the generated HTML for the table of contents
print("\nGenerated HTML for Table of Contents:")
print(toc_html)
