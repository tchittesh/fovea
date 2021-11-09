grid_generator = dict(
    type='PlainKDEGrid',
    output_shape=(600, 960),
    separable=True,
    attraction_fwhm=4,
    amplitude_scale=1,
    bandwidth_scale=64,
    anti_crop=True
)
