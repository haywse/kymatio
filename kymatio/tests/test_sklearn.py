import os
import torch
import io
import numpy as np

from kymatio.sklearn import ScatteringTransformer
from kymatio.scattering2d import Scattering2D

def test_sklearn_transformer():
    test_data_dir = os.path.join(os.path.dirname(__file__), "..",
                                 "scattering2d", "tests")

    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buf = io.BytesIO(f.read())
        data = np.load(buf)

    x = torch.from_numpy(data['x'])
    J = data['J']

    S = Scattering2D(J, x.shape[2:], frontend='torch')
    Sx = S.forward(x)

    x_raveled = x.reshape(x.shape[0], -1).detach().cpu().numpy()
    Sx_raveled = Sx.reshape(x.shape[0], -1).detach().cpu().numpy()

    st = ScatteringTransformer(S, x[0].shape,'torch').fit()

    t = st.transform(x_raveled)
    assert np.allclose(Sx_raveled, t)

    # Check numpy
    S = Scattering2D(J, x.shape[2:], frontend='numpy')
    st = ScatteringTransformer(S, x[0].shape, 'numpy').fit()

    t = st.transform(x_raveled)

    assert np.allclose(Sx_raveled, t)
