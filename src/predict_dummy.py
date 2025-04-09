from enums import AnimalType

def predict(prompt: str, animal_type: AnimalType) -> str:
    """
    Dummy predict function that simulates image generation.

    Args:
        prompt (str): The prompt describing the desired image.
        animal_type (AnimalType): The type of animal being requested.

    Returns:
        str: A constant path to a dummy image.
    """
    return "images/styled_generated_image_0.png"