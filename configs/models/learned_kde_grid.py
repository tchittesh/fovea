grid_generator = dict(
    type='LearnedKDEGrid',
    amplitude_init=1,
    bandwidth_init=64,
    output_shape=(600, 960),
    separable=True,
    attraction_fwhm=4,
    anti_crop=True
)
