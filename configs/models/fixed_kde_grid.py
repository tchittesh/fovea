grid_generator = dict(
    type='FixedKDEGrid',
    saliency_file='temporary',  # this will get dynamically replaced
    output_shape=(600, 960),
    separable=True,
    attraction_fwhm=4,
    anti_crop=True
)
