-- Query the five most recent scrobbles by play_time
SELECT * FROM scrobble ORDER BY play_time DESC LIMIT 5;

-- Query the number and ratio of 0/non-0 values in scrobble
SELECT
  'artist_mbid' AS column_name,
  COUNT(*) AS total_rows,
  COUNT(CASE WHEN artist_mbid IS NOT NULL AND TRIM(artist_mbid) != '' THEN 1 END) AS non_null_count,
  COUNT(CASE WHEN artist_mbid IS NULL OR TRIM(artist_mbid) = '' THEN 1 END) AS null_count,
  ROUND(COUNT(CASE WHEN artist_mbid IS NOT NULL AND TRIM(artist_mbid) != '' THEN 1 END) / COUNT(*) * 100, 2) AS non_null_percentage,
  ROUND(COUNT(CASE WHEN artist_mbid IS NULL OR TRIM(artist_mbid) = '' THEN 1 END) / COUNT(*) * 100, 2) AS null_percentage
FROM scrobble
UNION ALL
SELECT
  'album_title',
  COUNT(*),
  COUNT(CASE WHEN album_title IS NOT NULL AND TRIM(album_title) != '' THEN 1 END),
  COUNT(CASE WHEN album_title IS NULL OR TRIM(album_title) = '' THEN 1 END),
  ROUND(COUNT(CASE WHEN album_title IS NOT NULL AND TRIM(album_title) != '' THEN 1 END) / COUNT(*) * 100, 2),
  ROUND(COUNT(CASE WHEN album_title IS NULL OR TRIM(album_title) = '' THEN 1 END) / COUNT(*) * 100, 2)
FROM scrobble
UNION ALL
SELECT
  'track_title',
  COUNT(*),
  COUNT(CASE WHEN track_title IS NOT NULL AND TRIM(track_title) != '' THEN 1 END),
  COUNT(CASE WHEN track_title IS NULL OR TRIM(track_title) = '' THEN 1 END),
  ROUND(COUNT(CASE WHEN track_title IS NOT NULL AND TRIM(track_title) != '' THEN 1 END) / COUNT(*) * 100, 2),
  ROUND(COUNT(CASE WHEN track_title IS NULL OR TRIM(track_title) = '' THEN 1 END) / COUNT(*) * 100, 2)
FROM scrobble
UNION ALL
SELECT
  'artist_name',
  COUNT(*),
  COUNT(CASE WHEN artist_name IS NOT NULL AND TRIM(artist_name) != '' THEN 1 END),
  COUNT(CASE WHEN artist_name IS NULL OR TRIM(artist_name) = '' THEN 1 END),
  ROUND(COUNT(CASE WHEN artist_name IS NOT NULL AND TRIM(artist_name) != '' THEN 1 END) / COUNT(*) * 100, 2),
  ROUND(COUNT(CASE WHEN artist_name IS NULL OR TRIM(artist_name) = '' THEN 1 END) / COUNT(*) * 100, 2)
FROM scrobble
UNION ALL
SELECT
  'play_time',
  COUNT(*),
  COUNT(CASE WHEN play_time IS NOT NULL THEN 1 END),
  COUNT(CASE WHEN play_time IS NULL THEN 1 END),
  ROUND(COUNT(CASE WHEN play_time IS NOT NULL THEN 1 END) / COUNT(*) * 100, 2),
  ROUND(COUNT(CASE WHEN play_time IS NULL THEN 1 END) / COUNT(*) * 100, 2)
FROM scrobble;

-- album_title values with a length less than 2 characters
SELECT
  album_title,
  COUNT(*) AS count
FROM scrobble
WHERE LENGTH(TRIM(album_title)) < 2
GROUP BY album_title;

# Query unique artist name variants containing a specified string
SELECT DISTINCT artist_name FROM scrobble WHERE artist_name LIKE '%anna%' GROUP BY artist_name;

# Query certain the scrobble counts of given artists, e. g. before a festival (this could also be functionized)
SELECT
  CASE
    WHEN artist_name LIKE '%Faixa%' THEN 'Faixa'
    WHEN artist_name LIKE '%Anna of the North%' THEN 'Anna of the North'
    WHEN artist_name LIKE '%Yellow Days%' THEN 'Yellow Days'
    WHEN artist_name LIKE '%Kruder & Dorfmeister%' THEN 'Kruder & Dorfmeister'
    WHEN artist_name LIKE '%Mr Sanchez%' THEN 'Mr Sanchez'
    WHEN artist_name LIKE '%Deadletter%' THEN 'Deadletter'
    WHEN artist_name LIKE '%Teenage Fanclub%' THEN 'Teenage Fanclub'
    WHEN artist_name LIKE '%The Vaccines%' THEN 'The Vaccines'
    WHEN artist_name LIKE '%Death in Vegas%' THEN 'Death in Vegas'
    ELSE 'Other'
  END AS canonical_artist_name,
  COUNT(*) AS scrobble_count
FROM scrobble
WHERE
  artist_name LIKE '%Faixa%' OR
  artist_name LIKE '%Anna of the North%' OR
  artist_name LIKE '%Yellow Days%' OR
  artist_name LIKE '%Kruder & Dorfmeister%' OR
  artist_name LIKE '%Mr Sanchez%' OR
  artist_name LIKE '%Deadletter%' OR
  artist_name LIKE '%Teenage Fanclub%' OR
  artist_name LIKE '%The Vaccines%' OR
  artist_name LIKE '%Death in Vegas%'
GROUP BY canonical_artist_name
ORDER BY scrobble_count DESC;

# Query top artists from table scrobble
SELECT
    ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
    artist_name,
    COUNT(*) AS play_count
FROM
    scrobble
GROUP BY
    artist_name
ORDER BY
    play_count DESC
LIMIT 10;

# Query top artists with comma from table scrobble
WITH ranked_artists AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
        artist_name,
        COUNT(*) AS play_count
    FROM
        scrobble
    GROUP BY
        artist_name
)
SELECT
    rank_number,
    artist_name,
    play_count
FROM
    ranked_artists
WHERE
    artist_name LIKE '%,%'
ORDER BY
    rank_number
LIMIT 10;

# Query top albums with comma from table scrobble
WITH ranked_albums AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
        album_title,
        COUNT(*) AS play_count
    FROM
        scrobble
    GROUP BY
        album_title
)
SELECT
    rank_number,
    album_title,
    play_count
FROM
    ranked_albums
WHERE
    album_title LIKE '%,%'
ORDER BY
    rank_number
LIMIT 10;

# Query top tracks with comma from table scrobble
WITH ranked_tracks AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
        track_title,
        COUNT(*) AS play_count
    FROM
        scrobble
    GROUP BY
        track_title
)
SELECT
    rank_number,
    track_title,
    play_count
FROM
    ranked_tracks
WHERE
    track_title LIKE '%,%'
ORDER BY
    rank_number
LIMIT 10;

# Artists with a @
WITH ranked_artists AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
        artist_name,
        COUNT(*) AS play_count
    FROM
        scrobble
    GROUP BY
        artist_name
)
SELECT
    rank_number,
    artist_name,
    play_count
FROM
    ranked_artists
WHERE
    artist_name LIKE '%@%'
ORDER BY
    rank_number;

# Artists with a {
WITH ranked_artists AS (
    SELECT
        ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
        artist_name,
        COUNT(*) AS play_count
    FROM
        scrobble
    GROUP BY
        artist_name
)
SELECT
    rank_number,
    artist_name,
    play_count
FROM
    ranked_artists
WHERE
    artist_name LIKE '%{%'
ORDER BY
    rank_number;

# Query play_time values with a semicolon
SELECT
    ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC) AS rank_number,
    play_time,
    COUNT(*) AS play_time_count
FROM
    scrobble
WHERE
    play_time LIKE '%;%'
GROUP BY
    play_time
ORDER BY
    play_time_count DESC
LIMIT 10;

# Kveeri, ami megbizonyítja, hogy kiadja, hogy hány milyen-ASCII-s előadónévből mennyi van
SELECT
    ac.ascii_char,
    COUNT(DISTINCT s.artist_name) AS unique_artist_count
FROM
    ascii_chars ac
LEFT JOIN
    scrobble s
    ON s.artist_name LIKE
        CONCAT('%', REPLACE(REPLACE(REPLACE(ac.ascii_char, '\\', '\\\\'), '%', '\\%'), '_', '\\_'), '%')
        ESCAPE '\\'
GROUP BY
    ac.ascii_char
ORDER BY
    unique_artist_count DESC;
