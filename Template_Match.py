def tamplate_match(in1, in2, mode='full', method='auto'):
    r"""
    Cross-correlate two N-dimensional arrays.
    Second input. Should have the same number of dimensions as in1. 
    Returns An N-dimensional array containing a subset of the discrete linear cross-correlation of in1 with in2.
    
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are 'valid',"
                         " 'same', or 'full'.") from e

    # this either calls fftconvolve or this function with method=='direct'
    if method in ('fft', 'auto'):
        return convolve(in1, _reverse_and_conj(in2), mode, method)

    elif method == 'direct':
        # fastpath to faster numpy.correlate for 1d inputs when possible
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        # _correlateND is far slower when in2.size > in1.size, so swap them
        # and then undo the effect afterward if mode == 'full'.  Also, it fails
        # with 'valid' mode if in2 is larger than in1, so swap those, too.
        # Don't swap inputs for 'same' mode, since shape of in1 matters.
        swapped_inputs = ((mode == 'full') and (in2.size > in1.size) or
                          _inputs_swap_needed(mode, in1.shape, in2.shape))

        if swapped_inputs:
            in1, in2 = in2, in1

        if mode == 'valid':
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            out = np.empty(ps, in1.dtype)

            z = _sigtools._correlateND(in1, in2, out, val)

        else:
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]

            # zero pad input
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == 'full':
                out = np.empty(ps, in1.dtype)
            elif mode == 'same':
                out = np.empty(in1.shape, in1.dtype)

            z = _sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # Reverse and conjugate to undo the effect of swapping inputs
            z = _reverse_and_conj(z)

        return z

    else:
        raise ValueError("Acceptable method flags are 'auto',"
                         " 'direct', or 'fft'.")
