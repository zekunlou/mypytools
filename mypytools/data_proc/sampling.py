import sys
import numpy


def do_fps_new(data:numpy.ndarray, select_cnt:int, seed=None) -> numpy.ndarray:
    """Furthest point sampling for data (shape [nsamples, nfeat])

    Returns:
        numpy.ndarray: the selected idxes (shape [select_cnt,])
    """

    assert isinstance(data, numpy.ndarray)
    assert data.ndim == 2, f"should have data.ndim == 2 by now, but {data.ndim=}"
    assert isinstance(select_cnt, int)
    if select_cnt >= data.shape[0]:
        print("WARNING: select_cnt >= data.shape[0], return all idxes", file=sys.stderr)
        return numpy.arange(data.shape[0])
    assert 0 < select_cnt < data.shape[0], f"should have 0 < select_cnt < data_size={data.shape[0]}, but {select_cnt=}"

    if seed is not None:
        assert isinstance(seed, int)
        rng = numpy.random.default_rng(seed)
        start_idx = rng.integers(0, data.shape[0])
    else:
        start_idx = 0
    select_idxes = [None for _ in range(select_cnt)]  # start from a random point
    select_idxes[0] = start_idx  # start from a random point

    data_square = numpy.sum(numpy.square(data).real, axis=-1)  # real, shape (nsamples,)
    """Trick: maintain an all-data to select-data minimum distance square array"""
    to_select_dist_square = data_square + data_square[start_idx] \
        - 2 * numpy.real(numpy.dot(data, data[start_idx].conj()))  # real, shape (nsamples, ...,)
        # - 2 * numpy.real(numpy.sum(data * data[start_idx], axis=-1))  # real, shape (nsamples, ...,)

    for i in range(1, select_cnt):  # evaluate the i-th point
        current_idx = select_idxes[i-1]
        current_data = data[current_idx]
        """calculate the distance square from current_data to all data"""
        to_current_dist_square = data_square + data_square[current_idx] \
            - 2 * numpy.real(numpy.dot(data, current_data))
            # - 2 * numpy.real(numpy.sum(data * current_data, axis=-1))
        """update the minimum distance square from all data to selected data"""
        to_select_dist_square = numpy.minimum(to_select_dist_square, to_current_dist_square)
        """select the next point"""
        next_idx = numpy.argmax(to_select_dist_square)
        select_idxes[i] = next_idx

    return numpy.array(select_idxes, dtype=int)
