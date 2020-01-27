"""
ScikitLearn Transformer
=====================================================================
Here we demonstrate a simple application of scattering as a transformer
"""


from kymatio import Scattering2D
from kymatio.utils import ScatteringTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

# Use numpy or torch for the scattering
frontend = 'numpy'


# Create a scattering object (can be Scattering1D or Scattering3D as well)
S = Scattering2D(shape = (8, 8), J = 1, frontend = frontend)

# Use the toy digits dataset (8x8 digits)
digits = datasets.load_digits()

# Split data into train and test subsets
X_train, X_test, y_train, y_test = train_test_split(
    digits.images.reshape((len(digits.images), -1)), digits.target, test_size=0.5, shuffle=False)

# Create the scikitlearn transformer
st = ScatteringTransformer(S, digits.images[0].shape,frontend)

# Create a classifier: a support vector classifier
classifier = LogisticRegression()
estimators = [('scatter', st), ('clf', classifier)]
pipe = Pipeline(estimators)

# We learn the digits on the first half of the digits
pipe.fit(X_train, y_train)

# Now predict the value of the digit on the second half:
y_pred = pipe.predict(X_test)
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))