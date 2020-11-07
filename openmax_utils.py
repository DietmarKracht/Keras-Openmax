import scipy.spatial.distance as spd


class DistanceType:

    def eucos(self):
        return 'eucos'

    def euclidean(self):
        return 'euclidean'

    def cosine(self):
        return 'cosine'

    def squeuclidean(self):
        return 'squeuclidean'


def compute_distance(query_channel, channel, mean_vec, distance_type = 'eucos'):
    """ Compute the specified distance type between chanels of mean vector and query image.
    In caffe library, FC8 layer consists of 10 channels. Here, we compute distance
    of distance of each channel (from query image) with respective channel of
    Mean Activation Vector. In the paper, we considered a hybrid distance eucos which
    combines euclidean and cosine distance for bouding open space. Alternatively,
    other distances such as euclidean or cosine can also be used.

    Input:
    --------
    query_channel: Particular FC8 channel of query image
    channel: channel number under consideration
    mean_vec: mean activation vector
    Output:
    --------
    query_distance : Distance between respective channels
    """

    if distance_type == 'eucos':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel) /200. + spd.cosine(mean_vec[channel, :], query_channel)
    elif distance_type == 'euclidean':
        query_distance = spd.euclidean(mean_vec[channel, :], query_channel) /200.
    elif distance_type == 'cosine':
        query_distance = spd.cosine(mean_vec[channel, :], query_channel)
    elif distance_type == DistanceType.squeuclidean():
        query_distance = spd.sqeuclidean(mean_vec[channel, :], query_channel)
    else:
        print("distance type not known: enter either of eucos, euclidean or cosine")
    return query_distance