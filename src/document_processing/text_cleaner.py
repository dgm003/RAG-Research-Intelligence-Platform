import re
import unicodedata
import ftfy


def clean_text(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)

    # Repair words split across PDF line breaks.
    text = re.sub(r"(?<=\w)-[ \t]*\n[ \t]*(?=\w)", "", text)

    # Join wrapped lines and normalize whitespace.
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Remove control characters, preserving whitespace.
    text = "".join(
        char for char in text
        if char in "\n\t" or unicodedata.category(char) != "Cc"
    )

    return text.strip()


if __name__ == "__main__":
    sample = (
        "scientific ﬁelds and distribu-\nted data. "
        "Publisherâ€™s note. Copyright: Â© 2021."
    )
    print(clean_text(sample))