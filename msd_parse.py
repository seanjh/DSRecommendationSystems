import json
import config
from pyspark.mllib.recommendation import Rating

# user, song, play count triplets
# http://labrosa.ee.columbia.edu/millionsong/tasteprofile
USERID_INDEX = 0
SONGID_INDEX = 1
PLAYCOUNT_INDEX = 2


def parse_line(line):
    parts = line.strip().split("\t")
    return (
        parts[USERID_INDEX],
        parts[SONGID_INDEX],
        int(parts[PLAYCOUNT_INDEX])
    )


def rating_convert(row):
    return Rating(
        int(row[USERID_INDEX]),
        int(row[SONGID_INDEX]),
        int(row[PLAYCOUNT_INDEX])
    )


def make_id_map(ids_rdd):
    ids_ints = ids_rdd.distinct().zipWithUniqueId()
    return dict(ids_ints.collect())


def make_users_map(rdd):
    users_map = make_id_map(rdd.map(lambda row: row[USERID_INDEX]))
    with open(config.MSD_USERID_MAP, 'w') as outfile:
        json.dump(users_map, outfile)
    return users_map


def make_songs_map(rdd):
    songs_map = make_id_map(rdd.map(lambda row: row[SONGID_INDEX]))
    with open(config.MSD_SONGID_MAP, 'w') as outfile:
        json.dump(songs_map, outfile)
    return songs_map


def make_row_coverter(users, songs):
    def convert_row(row):
        return (
            int(users.get(str(row[USERID_INDEX]))),
            int(songs.get(str(row[SONGID_INDEX]))),
            row[PLAYCOUNT_INDEX]
        )
    return convert_row


def replace_raw_ids(train, users, songs):
    convert_row_func = make_row_coverter(users, songs)
    return train.map(convert_row_func)


# convert user_id, song_id to integer
