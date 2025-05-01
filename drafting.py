
top10 = data["Artist"].value_counts().head(10)
top10.barplot(x=top10.values, y=top10.index, palette="husl", legend=False)
top10.title("Top 10 Artists by Scrobbles")
top10.tight_layout()
top10.show()

