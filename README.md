# Play it Again, Sam!

This is the official repository of [Giovanni Gabbolini](https://giovannigabbolini.github.io) and [Derek Bridge](http://www.cs.ucc.ie/~dgb/)'s paper "Play It Again, Sam! Recommending Familiar Music in Fresh Ways".

It is possible to replicate all the results shown in the paper by running: `src/tfp/paper.py`

## Installation

### Step 1

Create an environment with Python 3.9.2 and install dependencies by: `pip install -r requirements.txt`.
Also run `config.sh`.

### Step 2

Download the required data.
In particular, download and extract in the folder `res/r` the following data:

- Spotify's MPD: [https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge](https://www.aicrowd.com/challenges/spotify-million-playlist-dataset-challenge).

### Step 3

Prepare the required data by running: `config.py`.
