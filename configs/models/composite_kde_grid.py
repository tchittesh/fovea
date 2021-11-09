grid_generator = dict(
    type='CompositeKDEGrid',
    alpha=0.5,
    fixed_grid=dict(
        type='FixedKDEGrid',
        saliency_file='temporary',  # this will get dynamically replaced
    ),
    adaptive_grid=dict(
        type='PlainKDEGrid',
        amplitude_scale=1,
        bandwidth_scale=64,
    ),
    output_shape=(600, 960),
    separable=True,
    attraction_fwhm=4,
    anti_crop=True,
)
