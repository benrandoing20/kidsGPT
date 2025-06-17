from gutenbergpy.textget import get_text_by_id

kids_ids = [35, 36, 98]  # Alice in Wonderland, Peter Rabbit, etc.

with open("scrape/gutenberg_kids.txt", "w", encoding="utf-8") as out:
    for gid in kids_ids:
        raw = get_text_by_id(gid)
        clean = raw.decode("utf-8").split("*** START")[1].split("*** END")[0]
        out.write(clean + "\n")
