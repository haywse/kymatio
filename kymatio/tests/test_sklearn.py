import os
import io
import numpy as np

from kymatio.sklearn import ScatteringTransformer
from kymatio.numpy import Scattering2D

def test_sklearn_transformer():
    test_data_dir = os.path.join(os.path.dirname(__file__), "..",
                                 "scattering2d", "tests")

    with open(os.path.join(test_data_dir, 'test_data_2d.npz'), 'rb') as f:
        buf = io.BytesIO(f.read())
        data = np.load(buf)

    x = data['x']
    J = data['J']

    S = Scattering2D(J, x.shape[2:])
    Sx = S.scattering(x)

    x_raveled = x.reshape(x.shape[0], -1)
    Sx_raveled = Sx.reshape(x.shape[0], -1)

    st = ScatteringTransformer(S, x[0].shape).fit()

    t = st.transform(x_raveled)
    assert np.allclose(Sx_raveled, t)
