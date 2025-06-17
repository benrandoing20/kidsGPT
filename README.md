# kidsGPT

**kidsGPT** is an experimental project for training GPT models using lower-complexity English language data, ideal for children or language learners. The project collects and processes text from sources like Project Gutenberg, Simple English Wikipedia, and Storybooks.

---

## Getting Started (Source Only)

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd kidsGPT
```

### 2. Install [UV](https://github.com/astral-sh/uv)

If you don't have UV installed, run:

```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

### 3. Set Up the Python Environment

From the project root, run:

```bash
uv venv
uv sync
```

This will create a virtual environment and install all dependencies as specified in `pyproject.toml` and `uv.lock`.

---

## Running the Scrapers

Run each scraper from source to collect the data:

```bash
uv pip install .  # (if needed, to make sure dependencies are available)

uv run python scrape/gutenberg_scraper.py
uv run python scrape/simplewiki_scraper.py
uv run python scrape/storybooks_scraper.py
```

Each script will output a text file in the `scrape/` directory:
- `scrape/gutenberg_kids.txt`
- `scrape/simplewiki_kids.txt`
- `scrape/storybooks_kids.txt`

---

## Combine All Data into a Single File

After running all scrapers, combine the data into one file:

```bash
cat scrape/gutenberg_kids.txt scrape/simplewiki_kids.txt scrape/storybooks_kids.txt > all_kids_corpus.txt
```

This will create `all_kids_corpus.txt` in your project root, containing the full dataset.

---

## License

MIT License (or specify your license here)
