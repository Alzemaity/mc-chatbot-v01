import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import json

def scrape_website(base_url: str, max_pages=50):
    visited = set()
    to_visit = [base_url]
    texts = []

    while to_visit and len(visited) < max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = requests.get(url, verify=False, timeout=10)
            soup = BeautifulSoup(resp.text, "html.parser")

            # ✅ extract text
            page_texts = [p.get_text(strip=True) for p in soup.find_all("p")]
            for t in page_texts:
                if len(t) > 30:
                    texts.append(t)

            # ✅ add internal links
            for a in soup.find_all("a", href=True):
                link = urljoin(base_url, a["href"])
                if base_url in link and link not in visited:
                    to_visit.append(link)

        except Exception as e:
            print(f"⚠️ Error scraping {url}: {e}")

    return texts

def chunk_text(text, chunk_size=200):
    """Split text into smaller chunks for better retrieval"""
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

if __name__ == "__main__":
    base_url = "http://mc2.mc.edu.eg/"
    docs = scrape_website(base_url)

    # chunking
    chunks = []
    for doc in docs:
        chunks.extend(chunk_text(doc))

    # save to json
    with open("data.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"✅ Scraping done. {len(chunks)} chunks saved to data.json")
