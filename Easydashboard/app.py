from utils import plot_confusion_matrix, load_map, plot_SHAP, create_map
import hydralit as hy
import streamlit as st
import streamlit.components.v1 as components


st.set_option("deprecation.showPyplotGlobalUse", False)
app = hy.HydraApp(
    title="Explainer Dashboard",  # nav_container=st.header,
    nav_horizontal=False,
    navbar_animation=True,
    hide_streamlit_markers=True,
    use_navbar=True,
    navbar_sticky=False,
)

# Remove whitespace from the top of the page and sidebar
st.markdown(
    """
        <style>
               .css-18e3th9 {
                    padding-top: 0rem;
                    padding-bottom: 10rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
               .css-1d391kg {
                    padding-top: 3.5rem;
                    padding-right: 1rem;
                    padding-bottom: 3.5rem;
                    padding-left: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True,
)
hide_streamlit_style = """
            <style>
            [theme]
base="light"
primaryColor="#1e84d4"
font="serif"

            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.sidebar.title("Explainer Dashboard")

@app.addapp(title="About")
def about():
    c1, c2, c3 = st.columns((1, 6, 1))

    with c2:
        st.header('Need of Dashboard')
        st.write('As we transition from team to Squads, there will be occurences to discuss model performance and workings with the team.')
        st.write('Rather than sharing screenshots/plots of the model, dashboard could kindle collaborative efforts and speed up delivery')
        st.header('Target Audience')
        st.write('Target audience is us, developers. This is different from MPR which targets program performance.')
        st.header('Features')
        st.write('Threshold Adjustment and decision')
        st.write('SHAP Analysis at index level')
        st.write('Spatial Analysis')
        st.write('Online Parameter Training, may be?')
        st.header('Usage')
        st.write('pip install easydashboard and then run() to launch it inside any system')
        


@app.addapp(title="Classification Metrics")
def home():

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

    plot_confusion_matrix(data_type, threshold)


@app.addapp(title="SHAP Analysis")
def shap_analysis():
    data_type = st.sidebar.selectbox(
        "Select Test or train to see the metrics", ["Test", "Train"], index=0
    )
    c1, c2, c3 = st.columns((1, 6, 1))

    with c2:
        plot_SHAP(data_type)

@app.addapp(title="Create Spatial View")
def create_map_view():
       
    c1, c2, c3 = st.columns((1, 8, 1))

    with c2:
        st.title("Spatial view of predictions")
        components.html(create_map(), height=800)



@app.addapp(title="View Existing Map")
def spatial_view():
    c1, c2, c3 = st.columns((1, 6, 1))

    with c2:
        st.title("Spatial view")
        components.html(load_map(), height=600)


# Run the whole lot, we get navbar, state management and app isolation, all with this tiny amount of work.
app.run()

