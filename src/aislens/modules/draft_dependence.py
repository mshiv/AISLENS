import numpy as np
from sklearn.linear_model import LinearRegression

def dedraft(data, draft):
    """
    Remove draft dependence from the data using linear regression.

    Args:
        data (xarray.DataArray): Input data.
        draft (xarray.DataArray): Draft data.

    Returns:
        xarray.DataArray: Predicted draft dependence.
    """
    data_tm = data.mean(dim='Time')
    draft_tm = draft.mean(dim='Time')
    data_stack = data_tm.stack(z=('x', 'y'))
    draft_stack = draft_tm.stack(z=('x', 'y'))
    data_stack_noNaN = data_stack.fillna(0)
    draft_stack_noNaN = draft_stack.fillna(0)
    reg = LinearRegression().fit(draft_stack_noNaN.values.reshape(-1, 1), data_stack_noNaN.values.reshape(-1, 1))
    data_pred_stack_noNaN_vals = reg.predict(draft_stack_noNaN.values.reshape(-1, 1)).reshape(-1)
    data_pred_stack_noNaN = data_stack_noNaN.copy(data=data_pred_stack_noNaN_vals)
    data_pred_stack = data_pred_stack_noNaN.where(~data_stack.isnull(), np.nan)
    data_pred = data_pred_stack.unstack('z').transpose()
    return data_pred