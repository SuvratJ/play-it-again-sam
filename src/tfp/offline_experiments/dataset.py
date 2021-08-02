import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from src.data.data import raw_dataset_path, preprocessed_dataset_path
from src.knowledge_graph.construct_graph import construct_graph
from src.knowledge_graph.applicable_actions import InformativeActionSupplier


def prepare_dataset(folder_name, random_state, start_from=0):
    """Build a dataset starting from the Spotify's 2018 RecSys Challenge dataset.

    The playlists are sampled with stratified random sampling by playlist length, up to length 50, with buckets of size 20.
    So, 20 playlists of length 50, 20 playlists of length 49, ...
    The minimum playlist length is 5.
    So the dataset is of 920 playlists: 46 buckets of 20 playlists.

    Then, it builds song trees from the songs in the playlists we selected.
    The song trees are built with the informative part of the KG.
    """
    playlists = pd.read_csv(f"{raw_dataset_path}/spotify_recsys2018/playlists.csv", sep='\t', lineterminator='\r', usecols=['num_tracks', 'pid'])

    # shuffle
    playlists = playlists.sample(frac=1, random_state=random_state)

    bucket_size = 20
    buckets = [[] for _ in range(5, 51)]
    for i, desired_playlist_length in enumerate(range(5, 51)):
        for _, row in playlists.iterrows():

            if desired_playlist_length == row.num_tracks:
                buckets[i].append(int(row.pid))

            if len(buckets[i]) == bucket_size:
                break

    # build and save trees for the songs in the selected playlists
    del playlists
    interactions = pd.read_csv(f"{raw_dataset_path}/spotify_recsys2018/interactions.csv", sep='\t',
                               lineterminator='\r', usecols=['pid', 'tid'])
    tracks = pd.read_csv(f"{raw_dataset_path}/spotify_recsys2018/tracks.csv", sep='\t',
                         lineterminator='\r', usecols=['tid', 'arid', 'alid', 'track_name'])
    albums = pd.read_csv(f"{raw_dataset_path}/spotify_recsys2018/albums.csv", sep='\t', lineterminator='\r', usecols=['alid', 'album_name'])
    artists = pd.read_csv(f"{raw_dataset_path}/spotify_recsys2018/artists.csv", sep='\t', lineterminator='\r', usecols=['arid', 'artist_name'])

    nested_tree_seeds = []
    for _, bucket in enumerate(tqdm(buckets)):
        for pid in bucket:
            playlist = interactions[interactions['pid'] == pid]
            df = playlist.merge(tracks, how='left').merge(albums, how='left').merge(artists, how='left')

            tree_seeds = []
            for track_name, artist_name, album_name in zip(df.track_name, df.artist_name, df.album_name):
                d = {}
                # check that values are not nan
                if not pd.isnull(track_name):
                    d['track_name'] = track_name
                if not pd.isnull(artist_name):
                    d['artist_name'] = artist_name
                if not pd.isnull(album_name):
                    d['track_name'] = album_name
                tree_seeds.append(d)
            nested_tree_seeds.append(tree_seeds)

    os.makedirs(f"{preprocessed_dataset_path}/tfp/{folder_name}/", exist_ok=True)
    for playlist_number, tree_seeds in enumerate(tqdm(nested_tree_seeds)):
        if playlist_number >= start_from:
            trees = []
            for d in tree_seeds:
                g = construct_graph(d, supplier=InformativeActionSupplier())
                trees.append(g)
            np.save(f"{preprocessed_dataset_path}/tfp/{folder_name}/{playlist_number}", trees)
