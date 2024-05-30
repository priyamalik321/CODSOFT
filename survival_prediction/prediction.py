from sklearn.metrics import accuracy_score

class Predict:

    def __init__(self):
        pass


    def test_model(self, model_pipeline,   x_test,  y_test):

        y_pred = model_pipeline.predict(x_test)

        score = accuracy_score(y_test, y_pred)

        return y_pred,score


