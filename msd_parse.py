from pyspark.mllib.recommendation import Rating

# user, song, play count triplets
# http://labrosa.ee.columbia.edu/millionsong/tasteprofile
USERID_INDEX = 0
SONGID_INDEX = 1
PLAYCOUNT_INDEX = 2


def parse_line(line):
    parts = line.strip().split("::")
    return (
        parts[USERID_INDEX],
        parts[SONGID_INDEX],
        int(parts[PLAYCOUNT_INDEX])
    )


def rating_convert(row):
    return Rating(
        int(row[USERID_INDEX]),
        int(row[SONGID_INDEX]),
        float(row[PLAYCOUNT_INDEX])
    )


def convert_ids_to_int(parsed_rdd):
    user_ids_int = (
        parsed_rdd
        .map(lambda row: row[USERID_INDEX])
        .distinct()
        .zipWithUniqueId())
    print(user_ids_int.take(50))

    # song_ids_int = (
    #     parsed_rdd
    #     .map(lambda row: row[SONGID_INDEX])
    #     .distinct()
    #     .zipWithUniqueId())
    # print(song_ids_int.take(50))

# convert user_id, song_id to integer
