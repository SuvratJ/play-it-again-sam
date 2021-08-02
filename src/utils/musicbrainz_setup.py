import musicbrainzngs


def musicbrainzngs_setup():
    musicbrainzngs.set_useragent(
        "Sam",
        "0.1",
    )
    musicbrainzngs.set_rate_limit(limit_or_interval=False)
