import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import yaml
import shap

with open("./artifacts.yaml", "r") as stream:
    try:
        data = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        print(exc)


@st.cache()
def load_data(data_type):
    index_column = str(data["model"]["index"])
    if data_type == "Train":
        X = pd.read_csv(
            str('../' + data["model"]["X_train"]), low_memory=False, index_col=index_column
        )
        y = pd.read_csv(
            '../' + data["model"]["y_train"], low_memory=False, index_col=index_column
        )
    X = pd.read_csv('../' +data["model"]["X_test"], low_memory=False, index_col=index_column)
    y = pd.read_csv('../' +data["model"]["y_test"], low_memory=False, index_col=index_column)
    return X, y


def check_path(path_dict):
    for path in path_dict:
        if not isinstance(path, (str, type(None))):
            raise AttributeError("Data Path Must be String and Not NONE in .csv format")

    return


@st.cache(allow_output_mutation=True)
def load_model():
    path = '../' + data["model"]["model_path"]
    check_path(path)
    # pip install dill imbalanced-learn
    model = joblib.load(path)
    return model

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def plot_confusion_matrix(data_type="Test", threshold=0.5):
    from sklearn.metrics import (
        confusion_matrix,
        f1_score,
        roc_curve,
        precision_recall_curve,
        classification_report,
    )
    from numpy import sqrt, argmax, arange

    c1, c2, c3 = st.columns((1, 6, 1))

    # Load Model
    model = load_model()

    # Load Data
    X, y = load_data(data_type)

    # preds = model.predict(X)
    preds = (model.predict_proba(X)[:, 1] >= float(threshold)).astype(bool)
    cm = confusion_matrix(y, preds)

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout(pad=5.0, w_pad=5.0)
    sns.heatmap(cm, fmt="", annot=True, cmap="Blues", ax=axs[0])
    c2.markdown(
        "<h2 style='text-align: center;'>Confusion Matrix and Classification Report</h2>",
        unsafe_allow_html=True,
    )

    report = classification_report(y, preds, output_dict=True)

    sns.heatmap(
        pd.DataFrame(report).iloc[:-1, :].T, annot=True, ax=axs[1], cmap="Blues"
    )
    c2.pyplot(fig)

    # RCO and PR plot
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout(pad=5.0, w_pad=5.0)
    c2.markdown(
        "<h2 style='text-align: center;'>ROC Curve and P-R Curve</h2>",
        unsafe_allow_html=True,
    )

    # Roc Curve
    # predict probabilities
    yhat = model.predict_proba(X)
    # keep probabilities for the positive outcome only
    yhat = yhat[:, 1]
    # calculate roc curves
    fpr, tpr, thresholds = roc_curve(y, yhat)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1 - fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    st.sidebar.info(
        "ROC Curve Best Threshold=%f, G-Mean=%.3f" % (thresholds[ix], gmeans[ix])
    )
    # plot the roc curve for the model

    plt.clf()
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    fig.tight_layout(pad=5.0, w_pad=5.0)

    axs[0].plot([0, 1], [0, 1], linestyle="--", label="base")
    axs[0].plot(fpr, tpr, marker=".", label="Current Model")
    axs[0].scatter(fpr[ix], tpr[ix], marker="o", color="black", label="Best")
    # axis labels
    axs[0].set_xlabel("False Positive Rate")
    axs[0].set_ylabel("True Positive Rate")
    axs[0].legend()

    # PR Curve Plot
    precision, recall, thresholds = precision_recall_curve(y, yhat)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = argmax(fscore)
    st.sidebar.info(
        "PR Curve Best Threshold=%f, F-Score=%.3f" % (thresholds[ix], fscore[ix])
    )
    # plot the roc curve for the model
    no_skill = len(y[y == 1]) / len(y)

    axs[1].plot([0, 1], [no_skill, no_skill], linestyle="--", label="base")
    axs[1].plot(recall, precision, marker=".", label="Current Model")
    axs[1].scatter(recall[ix], precision[ix], marker="o", color="black", label="Best")
    # axis labels
    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].legend()

    c2.pyplot(fig)
    # plot_roc_curve(model, X, y, ax=axs[0])
    # plot_precision_recall_curve(model, X, y, ax=axs[1])

    # c2.pyplot(fig)
    with st.expander("See explanation"):
        st.write("ROC Thresholding")
        st.write(
            "'Youden's J statistic (also called Youden's index) is a single statistic that captures the performance of a dichotomous diagnostic test. Informedness is its generalization to the multiclass case and estimates the probability of an informed decision.'"
        )

    # apply threshold to positive probabilities to create labels
    def to_labels(pos_probs, threshold):
        return (pos_probs >= threshold).astype("int")

    # define thresholds
    thresholds = arange(0, 1, 0.001)
    # evaluate each threshold

    scores = [f1_score(y, to_labels(yhat, t)) for t in thresholds]
    # get best threshold
    ix = argmax(scores)
    st.sidebar.info(
        "Overall Best Threshold=%.3f, F-Score=%.5f" % (thresholds[ix], scores[ix])
    )

def create_map():
    # Load an empty map
    from keplergl import KeplerGl
    data_type = st.sidebar.selectbox(
        "Select Test or train to see the metrics", ["Test", "Train"], index=0
    )
    threshold = st.sidebar.slider(
        label="Prediction threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        format="%f",
    )
        
    X, y = load_data(data_type)
    model = load_model()
    preds = (model.predict_proba(X)[:, 1] >= float(threshold)).astype(bool)
    index_col = str(data['model']['index'])
    df = pd.concat([X, y],axis=1).reset_index()
    map_1 = KeplerGl(height=800, data={'Data': df, 
                                       'Predictions': pd.DataFrame({index_col: df[index_col],
                                                                    'predictions': preds})},
                     )
    return map_1._repr_html_()

@st.cache()
def load_map():
    if data["map"]["map_path"] is not None:
        HtmlFile = open('../' + data["map"]["map_path"], "r", encoding="utf-8")
        source_code = HtmlFile.read()
        return source_code
    else:
        return


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def create_explainer(X_sampled):
    model = load_model()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sampled)
    return explainer, shap_values


@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def st_shap(plot, height=None):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, height=height)


def plot_SHAP(data_type="Test"):

    # Load Data
    X, y = load_data(data_type)
    sample_size = int(data["shap"]["sample_size"])

    X_sampled = X.sample(n=sample_size)

    hid = st.sidebar.selectbox(
        label="Select Index to see its SHAP", options=X_sampled.index
    )

    explainer, shap_values = create_explainer(X_sampled)

    st.header("SHAP Force Plot (Stacked)")
    st_shap(
        shap.force_plot(explainer.expected_value[1], shap_values[1], X_sampled), 400
    )

    idx = X_sampled.loc[[hid]].reset_index().index.tolist()[0]
    st.header("SHAP Force Plot (Individual)")
    st_shap(
        shap.force_plot(
            explainer.expected_value[1], shap_values[1][idx, :], X_sampled.loc[hid]
        ),
        200,
    )

    st.header("SHAP Summary Plot")
    st.pyplot(shap.summary_plot(shap_values, X_sampled, plot_size=(5, 5)))
