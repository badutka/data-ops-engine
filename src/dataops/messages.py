COR_EX_001_MSG = "Found columns with number of unique values > 10. Consider dropping numerical or dense variables."

FEN_EX_001_MSG = "Argument 'encoder' must be 'ohe' or <None>."
FEN_EX_002_MSG = "Feature transformation resulted in sparse matrix, which was converted to numpy array. " \
                 "This may indicate high number of resulting feature columns, which can lead to slow processing. " \
                 "Make sure you are not using one hot encoder for columns with high number of unique categorical values."

MULTI_CLF_FIT_MSG_001 = "Preparing the fitting process for {} pipeline... ."
MULTI_CLF_FIT_MSG_002 = "Parameter grid for {} not provided. Running CV with libs' default parameters."
MULTI_CLF_FIT_MSG_003 = "Fitting data using {}... ."
MULTI_CLF_FIT_MSG_004 = "Fitting for {} completed successfully."
MULTI_CLF_FIT_MSG_005 = "Model parameters: {}"
MULTI_CLF_FIT_MSG_006 = "Fitting data using {}... ."
