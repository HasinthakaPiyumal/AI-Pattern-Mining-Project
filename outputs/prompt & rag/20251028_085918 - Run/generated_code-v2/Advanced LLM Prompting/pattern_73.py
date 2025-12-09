class TemplateManager:
    """Manages the creation, storage, and retrieval of question templates."""

    def __init__(self):
        self.templates = {}

    def add_template(self, template_name: str, template_string: str, placeholders: list):
        """Adds a new question template.

        Args:
            template_name (str): A unique name for the template.
            template_string (str): The template string with placeholders (e.g., "What is {concept} in {context}?").
            placeholders (list): A list of strings representing the expected placeholders in the template.
        """
        if not isinstance(template_name, str) or not template_name:
            raise ValueError("Template name must be a non-empty string.")
        if not isinstance(template_string, str) or not template_string:
            raise ValueError("Template string must be a non-empty string.")
        if not isinstance(placeholders, list) or not all(isinstance(p, str) for p in placeholders):
            raise ValueError("Placeholders must be a list of strings.")

        self.templates[template_name] = {
            "template_string": template_string,
            "placeholders": placeholders
        }
        print(f"Template '{template_name}' added successfully.")

    def get_template(self, template_name: str) -> dict or None:
        """Retrieves a template by its name.

        Args:
            template_name (str): The name of the template to retrieve.

        Returns:
            dict or None: The template dictionary if found, otherwise None.
        """
        return self.templates.get(template_name)

    def list_templates(self) -> list:
        """Lists all available template names.

        Returns:
            list: A list of template names.
        """
        return list(self.templates.keys())

    def update_template(self, template_name: str, new_template_string: str = None, new_placeholders: list = None):
        """Updates an existing template.

        Args:
            template_name (str): The name of the template to update.
            new_template_string (str, optional): The new template string. Defaults to None.
            new_placeholders (list, optional): The new list of placeholders. Defaults to None.
        """
        if template_name not in self.templates:
            raise ValueError(f"Template '{template_name}' not found.")

        if new_template_string is not None:
            if not isinstance(new_template_string, str) or not new_template_string:
                raise ValueError("New template string must be a non-empty string.")
            self.templates[template_name]["template_string"] = new_template_string

        if new_placeholders is not None:
            if not isinstance(new_placeholders, list) or not all(isinstance(p, str) for p in new_placeholders):
                raise ValueError("New placeholders must be a list of strings.")
            self.templates[template_name]["placeholders"] = new_placeholders

        print(f"Template '{template_name}' updated successfully.")

    def delete_template(self, template_name: str):
        """Deletes a template by its name.

        Args:
            template_name (str): The name of the template to delete.
        """
        if template_name in self.templates:
            del self.templates[template_name]
            print(f"Template '{template_name}' deleted successfully.")
        else:
            print(f"Template '{template_name}' not found.")