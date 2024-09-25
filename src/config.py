EVAL_BY_CONNECTIONS_PARAMS = {
    "min_connections": 0,
    "max_connections": 500,
    "bin_size": 10,
}

LABEL_PROPAGATION_PARAMS = {"alpha": 0.5, "num_iterations": 10, "keep_init": True}

CLASSES = {
    "ethnicity": ("hausa", "igbo", "yoruba"),
    "gender": ("m", "f"),
    "religion": ("christian", "muslim"),
}

SCORES_PATH = "predictions/"

NAMES_TO_ATTRIBUTES_MAP = "data/names_attribute_map.csv"

GENDER_SCRAPED_PATH = "data/names_gender_proportions.csv"

NM_RESULTS_PATH = "data/name_matching_eval.json"
