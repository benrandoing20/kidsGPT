import wikipediaapi
import re

wiki = wikipediaapi.Wikipedia('simple')

CATEGORIES = ["Children", "Basic English 850 words", "Science", "Animals", "Geography"]
MAX_PAGES = 100

def clean(text):
    return re.sub(r'\s+', ' ', text.strip()) # Replace white spaces with a single space in the stripped text

def get_pages(category):
    pages = []
    cat = wiki.page(f'Category:{category}')
    for title in cat.categorymembers:
        page = wiki.page(title)
        # Check if page exists and has substantial content
        if page.exists() and len(page.text.strip()) > 100:
            pages.append(clean(page.text))
        if len(pages) >= MAX_PAGES:
            break
    return pages

all_texts = []
for cat in CATEGORIES:
    print(f"Scraping category: {cat}")
    pages = get_pages(cat)
    all_texts.extend(pages)

with open("scrape/simplewiki_kids.txt", "w", encoding="utf-8") as f:
    for para in all_texts:
        f.write(para + "\n")
