from sklearn.preprocessing import PolynomialFeatures
from proj2_helpers import *
def poly_fit(X):
    """
    Fit the dataset using a polynomial basis.
    """
    poly = PolynomialFeatures(4, interaction_only=False)
    return poly.fit_transform(X)


def get_prediction_logreg(model, img,patch_size):
    IMG_PATCH_SIZE = patch_size
    data = np.asarray(img_crop(img, IMG_PATCH_SIZE, IMG_PATCH_SIZE))
    X = np.asarray([extract_features(data[i]) for i in range(len(data))])
    X_poly = poly_fit(X)
    output_prediction = model.predict(X_poly)
    img_prediction = label_to_img(img.shape[0], img.shape[1], IMG_PATCH_SIZE, IMG_PATCH_SIZE, output_prediction)

    return img_prediction
