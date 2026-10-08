import os
import re
import time
import argparse
from pathlib import Path

import requests
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

load_dotenv()

OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY")

if not OPENALEX_API_KEY:
    raise ValueError(
        "OPENALEX_API_KEY not found. "
        "Create a .env file and add your API key."
    )

BASE_API_URL = "https://api.openalex.org"
CONTENT_BASE_URL = "https://content.openalex.org/works"

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
METADATA_DIR = PROJECT_ROOT / "data" / "metadata"

METADATA_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Utility functions
# ---------------------------------------------------------

def clean_filename(text: str, max_length: int = 100) -> str:
    """
    Convert paper title into a filesystem-safe filename.
    """
    text = re.sub(r'[<>:"/\\|?*]', '', text)
    text = re.sub(r'\s+', '_', text.strip())

    return text[:max_length]


def get_work_id(openalex_id: str) -> str:
    """
    Convert:
        https://openalex.org/W123456
    into:
        W123456
    """
    return openalex_id.rstrip("/").split("/")[-1]


# ---------------------------------------------------------
# Search OpenAlex
# ---------------------------------------------------------

def search_openalex(
    query: str,
    count: int = 30,
    from_year: int = 2020,
):
    """
    Search OpenAlex for papers matching a topic/query.
    """

    url = f"{BASE_API_URL}/works"

    params = {
        "search": query,

        # We want open-access papers
        "filter": (
            f"open_access.is_oa:true,"
            f"has_content.pdf:true,"
            f"publication_year:>{from_year},"
            f"type:article"
        ),

        "per_page": min(count, 100),

        # Most relevant first
        "sort": "relevance_score:desc",

        "api_key": OPENALEX_API_KEY,
    }

    response = requests.get(url, params=params, timeout=30)

    response.raise_for_status()

    data = response.json()

    return data["results"][:count]


# ---------------------------------------------------------
# Download PDF
# ---------------------------------------------------------

def download_pdf(work, output_dir: Path):
    """
    Download and validate a research paper PDF.
    """

    openalex_id = get_work_id(work["id"])

    title = work.get("title") or "untitled"

    safe_title = clean_filename(title)

    filename = f"{openalex_id}_{safe_title}.pdf"

    output_path = output_dir / filename

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Already exists: {filename}")
        return True

    url = f"{CONTENT_BASE_URL}/{openalex_id}.pdf"

    params = {
        "api_key": OPENALEX_API_KEY
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=120,
            stream=True,
            headers={
                "User-Agent":
                    "ResearchIntelligencePlatform/1.0"
            }
        )

        response.raise_for_status()

        content_type = (
            response.headers
            .get("Content-Type", "")
            .lower()
        )

        print(
            f"\n{openalex_id}"
            f"\nContent-Type: {content_type}"
        )

        # Download the response
        with open(output_path, "wb") as f:

            for chunk in response.iter_content(
                chunk_size=8192
            ):

                if chunk:
                    f.write(chunk)

        # Validate file exists
        if not output_path.exists():

            print(
                f"ERROR: File was not created:"
                f" {output_path}"
            )

            return False

        file_size = output_path.stat().st_size

        # Validate non-empty
        if file_size == 0:

            print(
                f"ERROR: Empty file:"
                f" {output_path}"
            )

            output_path.unlink(
                missing_ok=True
            )

            return False

        # Validate PDF signature
        with open(
            output_path,
            "rb"
        ) as f:

            header = f.read(4)

        if header != b"%PDF":

            print(
                f"ERROR: Downloaded content"
                f" is not a valid PDF."
            )

            print(
                f"First bytes: {header!r}"
            )

            output_path.unlink(
                missing_ok=True
            )

            return False

        print(
            f"SUCCESS: {filename}"
            f" ({file_size / 1024 / 1024:.2f} MB)"
        )

        print(
            f"Saved to: {output_path}"
        )

        return True

    except requests.RequestException as e:

        print(
            f"\nFAILED: {openalex_id}"
            f"\n{e}"
        )

        return False


# ---------------------------------------------------------
# Save metadata
# ---------------------------------------------------------

def save_metadata(papers, topic):

    rows = []

    for paper in papers:

        best_location = paper.get(
            "best_oa_location"
        ) or {}

        rows.append({

            "paper_id":
                get_work_id(paper["id"]),

            "title":
                paper.get("title"),

            "publication_year":
                paper.get("publication_year"),

            "publication_date":
                paper.get("publication_date"),

            "doi":
                paper.get("doi"),

            "openalex_id":
                paper.get("id"),

            "topic_query":
                topic,

            "authors":
                "; ".join(
                    author["author"]["display_name"]
                    for author in paper.get(
                        "authorships", []
                    )
                    if author.get("author")
                ),

            "cited_by_count":
                paper.get("cited_by_count"),

            "is_open_access":
                paper.get("open_access", {}).get(
                    "is_oa"
                ),

            "oa_status":
                paper.get("open_access", {}).get(
                    "oa_status"
                ),

            "license":
                best_location.get("license"),

            "pdf_url":
                (
                    paper.get("content_urls", {})
                    or {}
                ).get("pdf"),

            "has_pdf":
                paper.get("has_content", {}).get(
                    "pdf",
                    False
                ),

            "has_grobid_xml":
                paper.get("has_content", {}).get(
                    "grobid_xml",
                    False
                ),

        })

    df = pd.DataFrame(rows)

    metadata_file = (
        METADATA_DIR /
        f"{clean_filename(topic)}_metadata.csv"
    )

    df.to_csv(
        metadata_file,
        index=False
    )

    print(
        f"\nMetadata saved to: {metadata_file}"
    )


# ---------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Download research papers from OpenAlex."
    )

    parser.add_argument(
        "--topic",
        required=True,
        help="Research topic to search for."
    )

    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Number of papers to download."
    )

    parser.add_argument(
        "--from-year",
        type=int,
        default=2020,
        help="Only retrieve papers after this year."
    )

    args = parser.parse_args()

    topic = args.topic

    print("\n====================================")
    print(" OpenAlex Research Paper Downloader")
    print("====================================")

    print(f"\nTopic: {topic}")
    print(f"Target papers: {args.count}")
    print(f"From year: {args.from_year}")

    # -----------------------------------------------
    # Step 1: Search OpenAlex
    # -----------------------------------------------

    print("\nSearching OpenAlex...")

    papers = search_openalex(
        query=topic,
        count=args.count,
        from_year=args.from_year
    )

    print(
        f"Found {len(papers)} candidate papers."
    )

    if not papers:
        print(
            "No papers found for this query."
        )
        return

    # -----------------------------------------------
    # Step 2: Create topic directory
    # -----------------------------------------------

    topic_dir = (
        RAW_DATA_DIR /
        clean_filename(topic).lower()
    )

    print(f"\nPDF destination:")
    print(topic_dir.resolve())

    print(f"\nMetadata destination:")
    print(METADATA_DIR.resolve())

    topic_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # -----------------------------------------------
    # Step 3: Download PDFs
    # -----------------------------------------------

    successful_downloads = 0

    print("\nDownloading PDFs...\n")

    for paper in tqdm(
        papers,
        desc="Downloading"
    ):

        success = download_pdf(
            paper,
            topic_dir
        )

        if success:
            successful_downloads += 1

        # Small delay between requests
        time.sleep(0.2)

    # -----------------------------------------------
    # Step 4: Save metadata
    # -----------------------------------------------

    save_metadata(
        papers,
        topic
    )

    # -----------------------------------------------
    # Summary
    # -----------------------------------------------

    print("\n====================================")
    print(" Download complete")
    print("====================================")

    print(
        f"Successful downloads: "
        f"{successful_downloads}/{len(papers)}"
    )

    print(
        f"PDF directory: {topic_dir}"
    )


if __name__ == "__main__":
    main()