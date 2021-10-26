from kedro.pipeline import Pipeline, node

from .nodes import independent_dependent_variable,concat_csv,preprocess_all_data,label_encoding,scaling,target_variable_skew



def create_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                func=independent_dependent_variable,
                inputs=["train","test"],
                outputs=["train_data", "test_data", "dependent_var"],
                name="independent_dependent_variable_node",
            ),
            node(
                func=concat_csv,
                inputs=["train_data", "test_data"],
                outputs="all_data",
                name="concat_csv_node",
            ),
            node(
                func=preprocess_all_data,
                inputs=["all_data"],
                outputs="preprocess_all_data",
                name="preprocess_all_data_node",
            ),
            node(
                func=label_encoding,
                inputs="preprocess_all_data",
                outputs="label_encoded_all_data",
                name="label_encoding_node",
            ),
            node(
                func=scaling,
                inputs="label_encoded_all_data",
                outputs="scaled_data",
                name="scaled_data_node",
            ),
            node(
                func=target_variable_skew,
                inputs="dependent_var",
                outputs="skewed_dependent_var",
                name="target_variable_skew_node",
            ),
        ]
    )