import random

def generate_architecture(task_description: str) -> str:
    """
    Simulates an LLM generating a neural network architecture string 
    based on a task description for an e-commerce recommender system.
    
    In a real scenario, an LLM would parse the task_description and 
    generate a suitable architecture description. Here, we use predefined 
    templates for demonstration.
    """
    architectures = [
        "InputLayer-EmbeddingLayer(item_dim=100, user_dim=100, embedding_size=64)-Dense(128)-Dense(64)-OutputLayer(activation=sigmoid)",
        "InputLayer-EmbeddingLayer(item_dim=150, user_dim=150, embedding_size=128)-Dense(256)-Dropout(0.3)-Dense(128)-OutputLayer(activation=sigmoid)",
        "InputLayer-CategoricalEmbedding(feature1=10, feature2=5)-Concatenate-Dense(96)-BatchNormalization-Dense(48)-OutputLayer(activation=sigmoid)",
        "InputLayer-SequentialDense(units=[256, 128, 64])-OutputLayer(activation=sigmoid)"
    ]
    
    # Simulate LLM choosing based on keywords (simplified)
    if "personalized" in task_description.lower():
        return random.choice(architectures[:2]) # Focus on embedding-based for personalization
    elif "feature interaction" in task_description.lower():
        return architectures[2] # Categorical embedding and concatenate
    else:
        return random.choice(architectures)

def parse_architecture_string(arch_string: str) -> dict:
    """
    Parses an architecture string into a dictionary representation.
    This is a simplified parser for demonstration.
    """
    layers = arch_string.split('-')
    parsed_layers = []
    for layer in layers:
        if '(' in layer and ')' in layer:
            name, params_str = layer.split('(', 1)
            params_str = params_str[:-1] # Remove closing parenthesis
            params = {}
            for p in params_str.split(','):
                if '=' in p:
                    key, value = p.split('=', 1)
                    params[key.strip()] = value.strip()
                else:
                    # Handle cases like Dropout(0.3) where it's just a value
                    params['value'] = p.strip()
            parsed_layers.append({'name': name.strip(), 'params': params})
        else:
            parsed_layers.append({'name': layer.strip(), 'params': {}})
    return {'layers': parsed_layers}

def architecture_to_string(arch_dict: dict) -> str:
    """
    Converts a dictionary representation of an architecture back to a string.
    """
    layers_str = []
    for layer in arch_dict['layers']:
        layer_name = layer['name']
        if layer['params']:
            param_parts = []
            if 'value' in layer['params'] and len(layer['params']) == 1:
                param_parts.append(str(layer['params']['value']))
            else:
                for k, v in layer['params'].items():
                    if k != 'value':
                        param_parts.append(f"{k}={v}")
            layers_str.append(f"{layer_name}({', '.join(param_parts)})")
        else:
            layers_str.append(layer_name)
    return '-'.join(layers_str)
