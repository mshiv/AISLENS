import numpy as np
import xarray as xr

from aislens.dataprep import simple_extrapolate_all_times


def test_simple_extrapolate_matches_ndimage():
	# small synthetic dataset
	nt = 2
	ny = 4
	nx = 5
	times = np.arange(nt)
	y = np.arange(ny)
	x = np.arange(nx)

	data = np.zeros((nt, ny, nx), dtype=float)
	data.fill(np.nan)
	# populate some valid points
	data[0, 1, 1] = 10.0
	data[0, 2, 3] = 20.0
	data[1, 0, 0] = 5.0
	data[1, 3, 4] = 15.0

	ds = xr.Dataset({'testvar': (('Time', 'y', 'x'), data)}, coords={'Time': times, 'y': y, 'x': x})

	out = simple_extrapolate_all_times(ds, 'testvar', time_dim='Time')

	assert 'testvar' in out
	filled = out['testvar'].values

	# No NaNs remain
	assert not np.isnan(filled).any()

	# The original source values must be preserved at their locations
	assert filled[0, 1, 1] == 10.0
	assert filled[0, 2, 3] == 20.0
	assert filled[1, 0, 0] == 5.0
	assert filled[1, 3, 4] == 15.0

	# Each time-slice should only contain the original non-NaN values (nearest-neighbour)
	vals_t0 = set(np.unique(filled[0]))
	src_t0 = set(np.unique(data[0][~np.isnan(data[0])]))
	assert vals_t0 == src_t0

	vals_t1 = set(np.unique(filled[1]))
	src_t1 = set(np.unique(data[1][~np.isnan(data[1])]))
	assert vals_t1 == src_t1
