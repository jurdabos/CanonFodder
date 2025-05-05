import questionary
from sqlalchemy import delete, update, select
from DB import SessionLocal
from DB.models import ArtistVariantsCanonized

SEPARATOR = "{"


def review_canonized_variants():
    """
    Interactive audit of the gold-standard table.
    """
    with SessionLocal() as sess:
        rows = sess.scalars(select(ArtistVariantsCanonized)).all()
        total = len(rows)
        print(f"\n📝  Reviewing {total} canonized variant groups…\n")
        for idx, row in enumerate(rows, 1):
            variants = row.artist_variants_text.split(SEPARATOR)
            print(f"\n[{idx}/{total}] ------------------------------------------------")
            print("Variants : ", " | ".join(variants))
            print("Canonical: ", row.canonical_name)
            print("to_link  : ", row.to_link)
            if row.comment:
                print("Comment  : ", row.comment)
            action = questionary.select(
                "Action?",
                choices=[
                    "Accept / next",
                    "Edit canonical",
                    "Toggle to_link",
                    "Add / edit comment",
                    "Delete row",
                    "Quit review"
                ]
            ).ask()
            if action == "Accept / next":
                continue
            elif action == "Edit canonical":
                new_can = questionary.text("New canonical name:").ask().strip()
                if new_can:
                    sess.execute(
                        update(ArtistVariantsCanonized)
                        .where(ArtistVariantsCanonized.artist_variants_hash == row.artist_variants_hash)
                        .values(canonical_name=new_can)
                    )
                    sess.commit()
            elif action == "Toggle to_link":
                sess.execute(
                    update(ArtistVariantsCanonized)
                    .where(ArtistVariantsCanonized.artist_variants_hash == row.artist_variants_hash)
                    .values(to_link=~row.to_link)
                )
                sess.commit()
            elif action == "Add / edit comment":
                new_comment = questionary.text(
                    "Enter comment (leave blank to clear):",
                    default=row.comment or ""
                ).ask()
                sess.execute(
                    update(ArtistVariantsCanonized)
                    .where(ArtistVariantsCanonized.artist_variants_hash == row.artist_variants_hash)
                    .values(comment=new_comment.strip() or None)
                )
                sess.commit()
            elif action == "Delete row":
                confirm = questionary.confirm("Really delete this entry?").ask()
                if confirm:
                    sess.execute(
                        delete(ArtistVariantsCanonized)
                        .where(ArtistVariantsCanonized.artist_variants_hash == row.artist_variants_hash)
                    )
                    sess.commit()
            elif action == "Quit review":
                print("👋  Exiting review early; changes so far are saved.")
                break
