import cdsapi

c = cdsapi.Client()
c.retrieve(
    'cams-global-reanalysis-eac4',
    {
        'date': '2009-01-01/2016-12-31',
        'format': 'netcdf',
        'variable': 'total_aerosol_optical_depth_550nm',
        'time': [
            '00:00', '03:00', '06:00',
            '09:00', '12:00', '15:00',
            '18:00', '21:00',
        ],
    },
    'data/cams/cams.nc')
