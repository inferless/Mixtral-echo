INPUT_SCHEMA = {
    "prompt": {
        'datatype': 'STRING',
        'required': True,
        'shape': [1],
        'example': ["What is deep meaning?"]
    },
    "temperature": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [0.7]
    },
    "repetition_penalty": {
        'datatype': 'FP32',
        'required': False,
        'shape': [1],
        'example': [1.18]
    },
    "max_new_tokens": {
        'datatype': 'INT16',
        'required': False,
        'shape': [1],
        'example': [128]
    }
}
