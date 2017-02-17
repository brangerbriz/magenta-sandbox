## Million Song Dataset Notes
[Paper](http://ismir2011.ismir.net/papers/OS6-1.pdf)

- Provides features and metadata to 1 Million popular contemporary wester commercial music tracks available to The Echo Nest.
- Called MSD for short
- Data:
	- 280 GB of data in MDF5 format (features and metadata only, no audio)
	- 1,000,000 Songs
	- 44,745 unique artists
	- 18,073 artists that have at least 20 songs in the dataset
	- 2,321 unique musicbrainz tags (only at the artist level)
	- 515,567 dated tracks starting from 1922
	- The main acoustic features are pitches, timbre and loudness, as defined by the Echo Nest Analyze AP for each "segment", which generally delineated by note onsets, or other discontinuities in the signal.
	- 237,662 tracks include lyrics (bag-of-words)
	- Even if you have the audio for a song that appears in the MSD, there is little guarantee that the features will have been computed on the same audio track.
- Example experiment in the paper is year prediction.
	- train/test split is at the artist level, not song level
	- the features used are the average and covariance of the timbre vectors for each song.
	- no further pre-processing is performed.
	- using only the non-redundent variables from the covariance matrix gives the feature vector of 90 elements per track.